import pandas as pd

langs = ["eng", "hin", "iben"]

for lang in langs:
    df = pd.read_csv(lang + "/trac2_{}_test.csv".format(lang))
    for task in ("a", "b"):
        preds = pd.read_csv("test_predictions{}_{}.txt".format(lang, task),
                            header=None)
        preds.columns = ["labels"]
        df["labels"] = preds["labels"]
        df = df[["ID", "labels"]]  # unique_id?
        df.to_csv("final_{}_{}.csv".format(lang, task), index=None)