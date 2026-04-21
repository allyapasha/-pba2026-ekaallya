# Scripts

Folder `scripts/` berisi utilitas operasional yang tidak perlu berada di root repository.

## Files

- `upload_to_hf_space.py`
  Script untuk upload folder `apps/hf_space/` ke Hugging Face Space.
- `sync_space_assets.py`
  Script untuk sinkronisasi artefak dari `artifacts/` ke folder paket deploy Hugging Face Space.

Wrapper root `upload_to_hf_space.py` tetap dipertahankan untuk kompatibilitas perintah lama.
