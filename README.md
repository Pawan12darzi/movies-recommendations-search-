# Movie Recommender GUI

A desktop movie browser and recommender built with Python, Tkinter, and
MovieLens data. Browse popular movies, filter by genre, search for similar
movies, and open a YouTube search for a movie from its card.

## Requirements

- Windows
- Python 3.10 or newer
- MovieLens `ml-latest-small` dataset

## Setup

1. Download and extract the
   [MovieLens latest-small dataset](https://grouplens.org/datasets/movielens/latest/).
2. Place the extracted `ml-latest-small` folder beside `movie_recommender.py`.
   Confirm that `ml-latest-small/movies.csv` is present. `ratings.csv` is
   optional, but is used for popularity and average ratings.
3. Open PowerShell in the project folder and run:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python movie_recommender.py
   ```

You can also start the app by double-clicking `run_movie_recommender.bat`
after setting up the Python environment.

The app creates its charts in `outputs/`. The dataset, virtual environment,
generated charts, and Python cache files are intentionally excluded from Git.
