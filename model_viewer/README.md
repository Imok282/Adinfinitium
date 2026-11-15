# model_viewer/ — 3D model viewer (trial)

Local Streamlit page (port 8504) that shows what our models predict, in 3D, on real NASA LOLA terrain:
terrain + hazard (U-Net vs slope-rule backup vs NASA's official map, with error maps), the skyline (light model vs exact vs backup, with real SPICE Sun/Earth directions),
and generated terrain versions (GENERATED). Runs the models on CPU so it does not compete with GPU training.

```bash
streamlit run model_viewer/app.py --server.port 8504
```
It is a harness for looking at predictions, not a contract-only screen: it imports `worldmodel.*` directly. Not a safety tool.
