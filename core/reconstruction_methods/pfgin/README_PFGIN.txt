PFGIN BENCHMARK
===============

Install the upstream version matching the original TPSSR snapshot:

    python tpssr.py setup-external

Then run:

    python tpssr.py prepare --from-folders
    python tpssr.py external --from-folders --method pfgin --stage both

PFGIN is trained on the training acquisitions and evaluated on the test acquisitions.
Source: https://github.com/yinhuwu/PFGIN
