Projeto IA 2024

```bash
pyenv install 3.8
pyenv shell 3.8
python -m venv marioenv
source marioenv/bin/activate
pip install gym==0.21.0
pip install gym-retro
cp rom.sfc marioenv/lib/python3.8/site-packages/retro/data/stable/SuperMarioWorld-Snes/
python train.py SHOW_GAME
```
