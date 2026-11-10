DDS2M BENCHMARK
===============

Install the upstream version matching the original TPSSR snapshot:

    python tpssr.py setup-external

Then run:

    python tpssr.py prepare --from-folders
    python tpssr.py external --from-folders --method dds2m --stage test

DDS2M optimizes each corrupted sample separately and may take a long time.
The local installation includes the `latest.mat` disk-space compatibility fix.
Source: https://github.com/miaoyuchun/DDS2M
