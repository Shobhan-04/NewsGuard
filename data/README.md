# Dataset instructions

**The dataset is not included in this repository.** Its redistribution terms were
not verified, so the CSV files are git-ignored. Download it yourself and place
the files in this folder.

## ISOT Fake News Dataset (default)

Expected files (names are matched case-insensitively):

| File | Label assigned by `train.py` | Columns |
|------|------------------------------|---------|
| `True.csv` | `REAL` (articles crawled from Reuters.com) | `title, text, subject, date` |
| `Fake.csv` | `FAKE` (articles from sites flagged by fact-checkers) | `title, text, subject, date` |

Neither file has a label column; the label comes from the file name.

**Where to get it**

1. The University of Victoria ISOT lab publishes the dataset. Start from the
   official page (`https://www.uvic.ca/engineering/ece/isot/datasets/index.php`; if it
   has moved, search for "ISOT Fake News Dataset University of Victoria").
2. The same files are widely mirrored on Kaggle. **Read the licence/terms on the
   page you download from** and follow any attribution the authors request.

I have not confirmed the licence text; check it before redistributing the data or
publishing derived datasets.

**Attribution**: Ahmed, H., Traore, I., Saad, S. (2017). *Detection of Online Fake
News Using N-Gram Analysis and Machine Learning Techniques.* In: Traore I., Woungang
I., Awad A. (eds) Intelligent, Secure, and Dependable Systems in Distributed and Cloud
Environments (ISDDC 2017), LNCS 10618, Springer, pp. 127-138.
DOI: 10.1007/978-3-319-69155-8_9

## How NewsGuard treats the data

* `subject` and `date` are **not** model features. The two files use different
  subject values, which would leak the label. `date` is read only for
  `python train.py --split time`.
* Reuters datelines (e.g. `WASHINGTON (Reuters) -`), the word "Reuters", URLs,
  @handles and a few source markers are removed (`preprocess.py`).
* Rows with missing/empty body, very short articles, exact duplicates and
  near-duplicates (same first 300 alphanumeric characters) are removed
  **before** splitting. Duplicates with conflicting labels are dropped entirely.

## Using your own CSV

```powershell
python train.py --csv data\my_articles.csv
```

Required columns: `text`, `label`. Optional: `title`, `date`. Accepted label
values (case-insensitive): `real`, `true`, `reliable`, `fake`, `false`, `unreliable`.

## Known dataset limitations

* Mostly English-language US political/world news from roughly 2015-2018.
* The classes come from different *sources*, so a model can learn source style
  rather than truthfulness.
* "FAKE" means "from a flagged outlet", not "every claim was checked and false".
