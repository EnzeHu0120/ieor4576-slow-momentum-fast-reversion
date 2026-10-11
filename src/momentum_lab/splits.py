"""Chronological folds and label-date purging, with validation-only model selection."""
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_start: str
    validation_start: str
    test_start: str
    test_end: str


def expanding_folds(train_start=1990, first_test=1995, end=2020, step=5):
    folds=[]
    for y in range(first_test,end,step):
        # Calendar 90/10 split, recorded explicitly; upstream per-ticker row split differs.
        start=pd.Timestamp(train_start,1,1); test=pd.Timestamp(y,1,1)
        val=(start+(test-start)*.9).normalize()
        folds.append(Fold(str(start.date()),str(val.date()),str(test.date()),
                          f"{min(y+step,end)}-01-01"))
    return folds


def masks(features, fold):
    t=pd.to_datetime(features["date"]); label=pd.to_datetime(features["label_date"])
    # The last train label cannot use a return belonging to validation or test.
    train=(t>=fold.train_start)&(t<fold.validation_start)&(label<fold.validation_start)
    val=(t>=fold.validation_start)&(t<fold.test_start)&(label<fold.test_start)
    test=(t>=fold.test_start)&(t<fold.test_end)&(label<fold.test_end)
    return train,val,test
