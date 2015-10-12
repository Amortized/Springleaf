import pickle;
import xgboost as xgb;
import pandas as pd;

model = xgb.Booster({'nthread': 8})
model.load_model('./data/1.xgb_model')


test    = pd.read_pickle("/mnt/data/Springleaf/test.csv.pickle");
test_id = test.ID;
test_X  = test.as_matrix();

t_x     = xgb.DMatrix(t_x, missing=float('NaN'));
y_hat   = model.predict(t_x,ntree_limit=930);

df = pd.DataFrame({'ID':test_id, 'target' : y_hat});
df.to_csv('submission.csv', index=False);


