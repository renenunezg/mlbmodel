import xgboost as xgb
import matplotlib.pyplot as plt

from backend.model import XGB_MODEL_PATH

model = xgb.XGBRegressor()
model.load_model(str(XGB_MODEL_PATH))

xgb.plot_importance(model)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()