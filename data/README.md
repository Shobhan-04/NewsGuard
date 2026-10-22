
## Where to get it

1. **Kaggle (download the dataset):** [Fake and Real News Dataset — ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. **Official University of Victoria ISOT page:** [ISOT Datasets](https://www.uvic.ca/engineering/ece/isot/datasets/index.php). If the page has moved, search for "ISOT Fake News Dataset University of Victoria".

### Download instructions

1. Open the [Kaggle dataset page](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
2. Sign in to Kaggle if prompted.
3. Download the dataset ZIP file and extract it.
4. Copy `True.csv` and `Fake.csv` into your project's `data/` directory.
5. Confirm the files are located at:

   ```text
   NewsGuard/
   └── data/
       ├── True.csv
       └── Fake.csv
   ```

6. Run training from the project root:

   ```powershell
   python train.py
   ```

**Licence and attribution:** The Kaggle listing identifies the dataset licence as **CC BY-NC-SA 4.0**. Review the licence terms on the download page before redistributing the dataset or derived datasets, and follow the authors' attribution requirements. The licence of the dataset does not automatically determine the licence of this project's code.
