# Apps

Folder `apps/` menyimpan entrypoint aplikasi dan paket deploy.

## Subfolders

- `local/`
  App lokal production yang dipakai untuk inferensi dan demo lokal.
- `hf_space/`
  Paket deploy production resmi untuk Hugging Face Space.
- `hf_space_deep_learning/`
  Paket deploy eksperimen terpisah untuk baseline non-production.

Jalur default untuk production adalah `apps/local/` dan `apps/hf_space/`.
