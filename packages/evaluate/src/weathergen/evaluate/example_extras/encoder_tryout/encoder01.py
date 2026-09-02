import time
from pathlib import Path

import torch

import weathergen.common.config as config
from weathergen.common.config import load_streams
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.model_interface import get_model, init_model_and_shard
from weathergen.train.utils import (
    VAL,
    cfg_keys_to_filter,
    filter_config_by_enabled,
    get_active_stage_config,
)

baseconfig = None#Path("/e/project1/e-ext-2025e01-128/buschow1/WeatherGenerator/config/config_forecasting.yml")
CHECKPOINT_RUN_ID = "atmofs03"

# 1) use the same internal config bootstrap as the real inference path,
# but without adding any CLI args: the config is created by merging the saved
# model config with the usual empty overrides, then assigning a run_id.
cf = config.load_merge_configs(
    private_home=None,
    from_run_id=CHECKPOINT_RUN_ID,
    mini_epoch=0,
    base=baseconfig,
)
cf = config.set_run_id(cf, "debug_encoder_run", False)

# 2) mimic the runtime fields normally set by Trainer.init_ddp / the CLI bootstrap
cf.world_size = 1
cf.rank = 0
cf.local_rank = 0
cf.with_ddp = False
cf.with_fsdp = False

# 3) match the normal CLI bootstrap: the sampler expects cf.data_loading.rng_seed to exist
cf.data_loading.rng_seed = int(time.time())

# 4) load stream definitions exactly as the project expects
cf.streams = load_streams(Path(cf.streams_directory))

# 5) build the stage configs exactly like Trainer.init() does before model creation.
# This is the root cause: the checkpoint loader expects the same filtered/derived
# training/validation/test configs the trainer creates at runtime.
cf.training_config = filter_config_by_enabled(cf.get("training_config"), cfg_keys_to_filter)
cf.validation_config = get_active_stage_config(
    cf.training_config, cf.get("validation_config", {}), cfg_keys_to_filter
)
cf.test_config = get_active_stage_config(
    cf.validation_config, cf.get("test_config", {}), cfg_keys_to_filter
)

training_cfg = cf.training_config
validation_cfg = cf.validation_config

if "output" not in cf.test_config:
    cf.test_config.output = {
        "num_samples": 0,
        "normalized_samples": False,
        "streams": None,
    }
if "forecast" not in cf.test_config:
    cf.test_config.forecast = {
        "time_step": "06:00:00",
        "num_steps": 2,
        "offset": 1,
        "policy": "fixed",
    }

# 6) create a very small validation sample set the same way inference uses it
# Important: OmegaConf stores datetime-like values via interpolation strings, not as raw numpy.datetime64 objects.
cf.test_config.samples_per_mini_epoch = 1
cf.test_config.shuffle = False
cf.test_config._start_date = "2023-07-12T00:00"
cf.test_config.start_date = "${datetime:2023-07-12T00:00}"
cf.test_config._end_date = "2023-12-31T00:00"
cf.test_config.end_date = "${datetime:2023-12-31T00:00}"
cf.test_config.output.num_samples = 1
cf.test_config.forecast.num_steps = 64

### stuff below here is basically copied from Trainer.inference()

# 7) build sampler using the merged real inference config
# The stage is "val" because the inference path uses validation/test config objects
# constructed from the training config.
dataset = MultiStreamDataSampler(cf, cf.test_config, stage=VAL)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=None,
    batch_sampler=None,
    shuffle=False,
    num_workers=0,
)

# 8) diagnostic check: construct the model once first, then immediately compare with the
# real trainer path below. This is only for debugging shape mismatches; the runtime path still
# uses init_model_and_shard() to load the checkpoint.
print("cf.ae_local_dim_embed:", cf.ae_local_dim_embed)
print("cf.ae_global_dim_embed:", cf.ae_global_dim_embed)
print("cf.ae_local_num_blocks:", cf.ae_local_num_blocks)
print("cf.ae_global_num_blocks:", cf.ae_global_num_blocks)
print("cf.fe_num_blocks:", cf.fe_num_blocks)
print("cf.streams_directory:", cf.streams_directory)

model_debug = get_model(cf, "inference", dataset, {})
embed_layers = [
    m for m in model_debug.encoder.embed_engine.embeds.values() if hasattr(m, "embed") and hasattr(m.embed, "weight")
]
if embed_layers:
    print("debug model encoder input dim:", embed_layers[0].embed.weight.shape)
else:
    print("debug model encoder input dim: no Linear embed.weight found")
print("debug model latent dim:", model_debug.latent_pre_norm.normalized_shape)
print("debug model forecast blocks:", len(model_debug.forecast_engine.fe_blocks) if model_debug.forecast_engine else 0)

model, model_params = init_model_and_shard(
    cf,
    dataset,
    CHECKPOINT_RUN_ID,
    -1,
    cf.test_config.training_mode,
    torch.device("cuda:0"),
    False,
    False,
    {},
)
model.eval()
device = torch.device("cuda:0")

# 9) take one batch and run only the encoder
batch = next(iter(loader))
batch.to_device(device)
source_batch = batch.get_source_samples()
print("model device:", next(model.parameters()).device)
print("source_batch device:", source_batch.get_device())

# accessing the actual input data
print(batch.get_source_samples().samples[0].streams_data["ERA5"].source_tokens_cells[0].shape)

with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
    enabled=True,
):
    with torch.no_grad():
        tokens_global, posteriors = model.encoder(model_params, source_batch)

print("tokens_global shape:", tokens_global.shape)
print("posteriors shape:", posteriors.shape)

# optional: construct model latent state exactly like forward()
latent = model.tokens_to_latent_state(
    model.latent_pre_norm(tokens_global) if hasattr(model, "latent_pre_norm") else None,
    tokens_global,
)
print(type(latent))
print(latent.register_tokens.shape if latent.register_tokens is not None else None)

# export the first 100 latent dimensions of the encoder output in a HEALPix-aware xarray
from astropy_healpix import healpy as hp
import numpy as np
import xarray as xr

# tokens_global is [n_samples, n_cells, latent_dim]
latent_vec = tokens_global[0].detach().cpu().numpy()#[:, :100]
level = cf.healpix_level
nside = 2**level
num_cells = 12 * nside**2

# the model uses nested HEALPix ordering, so we can attach cell center lon/lat coordinates
cell_ids = np.arange(num_cells)
lon_deg, lat_deg = hp.pix2ang(nside=nside, ipix=cell_ids, lonlat=True, nest=True)

# If the batch is sparse and does not fill all cells, keep the actual order present in tokens_global
if latent_vec.shape[0] != len(cell_ids):
    cell_ids = np.arange(latent_vec.shape[0])
    lon_deg, lat_deg = hp.pix2ang(nside=nside, ipix=cell_ids, lonlat=True, nest=True)

latent_xr = xr.DataArray(
    latent_vec,
    dims=("cell", "latent_dim"),
    coords={
        "cell": cell_ids,
        "lon": ("cell", lon_deg),
        "lat": ("cell", lat_deg),
    },
    name="latent",
)

out_path = Path("latent_space_full.nc")
latent_xr.to_netcdf(out_path)
print(f"Wrote latent state to {out_path} with shape {latent_xr.shape}")