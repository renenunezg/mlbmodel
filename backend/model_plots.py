import xgboost as xgb
import matplotlib.pyplot as plt

model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

xgb.plot_importance(model)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()