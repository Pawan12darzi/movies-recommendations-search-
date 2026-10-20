import os
import re
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "ml-latest-small")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

MOVIES_FILE = os.path.join(DATA_DIR, "movies.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")


def load_recommender_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MOVIES_FILE):
        raise FileNotFoundError(
            f"Movie data was not found:\n{MOVIES_FILE}\n\n"
            "Place the MovieLens dataset in the ml-latest-small folder."
        )

    movies = pd.read_csv(MOVIES_FILE)
    if os.path.exists(RATINGS_FILE):
        ratings = pd.read_csv(RATINGS_FILE)
    else:
        ratings = pd.DataFrame(
            columns=["userId", "movieId", "rating", "timestamp"]
        )

    movies = movies.drop_duplicates(subset="movieId").dropna(subset=["title"]).copy()
    movies["genres"] = movies["genres"].fillna("").astype(str)
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")
    movies["average_rating"] = 0.0
    movies["rating_count"] = 0
    if not ratings.empty and {"movieId", "rating"}.issubset(ratings.columns):
        rating_summary = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        movies["average_rating"] = (
            movies["movieId"].map(rating_summary["mean"]).fillna(0.0)
        )
        movies["rating_count"] = (
            movies["movieId"].map(rating_summary["count"]).fillna(0).astype(int)
        )

    genre_counts = (
        movies["genres"]
        .str.split("|")
        .explode()
        .loc[lambda values: values.ne("") & values.ne("(no genres listed)")]
        .value_counts()
    )

    plt.figure(figsize=(10, 6))
    genre_counts.head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 Movie Genres")
    plt.xlabel("Number of Movies")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_genres.png"))
    plt.close()

    if not ratings.empty and "rating" in ratings.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=ratings, x="rating")
        plt.title("Movie Ratings Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Number of Ratings")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "ratings_distribution.png"))
        plt.close()

    movies["features"] = (
        movies["title"].fillna("").astype(str).str.lower()
        + " "
        + movies["genres"].str.replace("|", " ", regex=False).str.lower()
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(movies["features"])
    similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)

    return movies, ratings, similarity_matrix


def find_recommendations(movie_name, number, movies, similarity_matrix):
    query = movie_name.strip().lower()
    if not query:
        return None, []

    titles = movies["title"].fillna("").astype(str).str.lower()
    matches = movies[titles.str.contains(re.escape(query), regex=True)]
    if matches.empty:
        return None, []

    movie_index = matches.index[0]
    actual_title = str(movies.loc[movie_index, "title"])
    row_position = movies.index.get_loc(movie_index)
    similarity_scores = similarity_matrix.getrow(row_position).toarray().ravel()
    similar_positions = similarity_scores.argsort()[::-1]

    recommendations = []
    for position in similar_positions:
        if position == row_position:
            continue

        row = movies.iloc[position]
        recommendations.append(
            (str(row["title"]), str(row["genres"]), float(similarity_scores[position]))
        )
        if len(recommendations) == number:
            break

    return actual_title, recommendations


class MovieRecommenderApp:
    def __init__(self, root, movies, ratings, similarity_matrix):
        self.root = root
        self.movies = movies
        self.ratings = ratings
        self.similarity_matrix = similarity_matrix

        self.root.title("CineMatch | Movie Recommender")
        self.root.geometry("1000x700")
        self.root.minsize(760, 560)
        self.root.configure(bg="#0b1020")

        self.current_movies = []
        self._configure_styles()
        self._build_layout()
        popular = self.movies.sort_values(
            ["rating_count", "average_rating"], ascending=False
        ).head(36)
        self._show_movies(
            [
                (
                    str(row["title"]),
                    str(row["genres"]),
                    float(row["average_rating"]),
                    None,
                )
                for _, row in popular.iterrows()
            ],
            "Popular movies",
            "Browse the most-rated movies or search for recommendations.",
        )

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#111a2e",
            foreground="#e8edf7",
            fieldbackground="#111a2e",
            rowheight=36,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Treeview.Heading",
            background="#1a2740",
            foreground="#9eacc4",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        style.map(
            "Treeview",
            background=[("selected", "#293b60")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "TCombobox",
            fieldbackground="#111a2e",
            background="#1a2740",
            foreground="#e8edf7",
            arrowcolor="#e8edf7",
        )

    def _build_layout(self):
        container = tk.Frame(self.root, bg="#0b1020", padx=36, pady=28)
        container.pack(fill="both", expand=True)

        header = tk.Frame(container, bg="#0b1020")
        header.pack(fill="x", pady=(0, 24))

        tk.Label(
            header,
            text="CINEMATCH",
            bg="#0b1020",
            fg="#9b8cff",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Find your next favorite movie.",
            bg="#0b1020",
            fg="#f3f5fb",
            font=("Segoe UI", 25, "bold"),
        ).pack(anchor="w", pady=(5, 3))
        tk.Label(
            header,
            text="Discover films with similar titles and genres.",
            bg="#0b1020",
            fg="#9eacc4",
            font=("Segoe UI", 11),
        ).pack(anchor="w")

        stats = tk.Frame(container, bg="#0b1020")
        stats.pack(fill="x", pady=(0, 22))
        self._stat_card(stats, f"{len(self.movies):,}", "MOVIES").pack(
            side="left", fill="x", expand=True, padx=(0, 10)
        )
        ratings_count = len(self.ratings)
        self._stat_card(stats, f"{ratings_count:,}", "USER RATINGS").pack(
            side="left", fill="x", expand=True, padx=(0, 10)
        )
        genre_count = (
            self.movies["genres"]
            .str.split("|")
            .explode()
            .loc[lambda values: values.ne("") & values.ne("(no genres listed)")]
            .nunique()
        )
        self._stat_card(stats, f"{genre_count}", "GENRES").pack(
            side="left", fill="x", expand=True
        )

        search_panel = tk.Frame(container, bg="#151e32", padx=20, pady=18)
        search_panel.pack(fill="x", pady=(0, 20))

        tk.Label(
            search_panel,
            text="SEARCH FOR A MOVIE",
            bg="#151e32",
            fg="#aab6cc",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        controls = tk.Frame(search_panel, bg="#151e32")
        controls.pack(fill="x")
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(
            controls,
            textvariable=self.search_var,
            bg="#0e1729",
            fg="#f3f5fb",
            insertbackground="#f3f5fb",
            relief="flat",
            font=("Segoe UI", 12),
            highlightthickness=1,
            highlightbackground="#293650",
            highlightcolor="#8e7cff",
        )
        self.search_entry.pack(side="left", fill="x", expand=True, ipady=11, padx=(0, 12))
        self.search_entry.bind("<Return>", self._search)

        tk.Label(
            controls,
            text="SHOW",
            bg="#151e32",
            fg="#aab6cc",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(0, 8))
        self.number_var = tk.StringVar(value="10")
        ttk.Combobox(
            controls,
            textvariable=self.number_var,
            values=("5", "10", "15", "20"),
            state="readonly",
            width=4,
            font=("Segoe UI", 11),
        ).pack(side="left", ipady=8, padx=(0, 12))

        self.genre_var = tk.StringVar(value="All genres")
        genres = sorted(
            {
                genre
                for movie_genres in self.movies["genres"]
                for genre in movie_genres.split("|")
                if genre and genre != "(no genres listed)"
            }
        )
        genre_filter = ttk.Combobox(
            controls,
            textvariable=self.genre_var,
            values=("All genres", *genres),
            state="readonly",
            width=15,
            font=("Segoe UI", 10),
        )
        genre_filter.pack(side="left", ipady=8, padx=(0, 12))
        genre_filter.bind("<<ComboboxSelected>>", self._on_genre_selected)

        tk.Button(
            controls,
            text="Get recommendations",
            command=self._search,
            bg="#7967f7",
            fg="#ffffff",
            activebackground="#8c7cff",
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            padx=20,
            pady=11,
        ).pack(side="left")

        results_header = tk.Frame(container, bg="#0b1020")
        results_header.pack(fill="x", pady=(0, 10))
        self.results_title = tk.Label(
            results_header,
            text="Popular movies",
            bg="#0b1020",
            fg="#f3f5fb",
            font=("Segoe UI", 15, "bold"),
        )
        self.results_title.pack(side="left")
        self.status_var = tk.StringVar(value="Enter a movie title to get started.")
        tk.Label(
            results_header,
            textvariable=self.status_var,
            bg="#0b1020",
            fg="#9eacc4",
            font=("Segoe UI", 9),
        ).pack(side="right")
        gallery = tk.Frame(container, bg="#0b1020")
        gallery.pack(fill="both", expand=True)
        self.gallery_canvas = tk.Canvas(
            gallery, bg="#0b1020", highlightthickness=0, borderwidth=0
        )
        scrollbar = ttk.Scrollbar(
            gallery, orient="vertical", command=self.gallery_canvas.yview
        )
        self.gallery_canvas.configure(yscrollcommand=scrollbar.set)
        self.gallery_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.gallery_frame = tk.Frame(self.gallery_canvas, bg="#0b1020")
        self.gallery_window = self.gallery_canvas.create_window(
            (0, 0), window=self.gallery_frame, anchor="nw"
        )
        self.gallery_frame.bind("<Configure>", self._update_scroll_region)
        self.gallery_canvas.bind(
            "<Configure>",
            lambda event: self.gallery_canvas.itemconfigure(
                self.gallery_window, width=event.width
            ),
        )
        self.gallery_canvas.bind("<MouseWheel>", self._scroll_gallery)
        self.gallery_canvas.bind(
            "<Button-4>", lambda event: self._scroll_gallery_direction(event, -1)
        )
        self.gallery_canvas.bind(
            "<Button-5>", lambda event: self._scroll_gallery_direction(event, 1)
        )

    def _scroll_gallery(self, event):
        self.gallery_canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _scroll_gallery_direction(self, event, direction):
        if event.widget is self.gallery_canvas:
            self.gallery_canvas.yview_scroll(direction, "units")
        return "break"

    def _update_scroll_region(self, event):
        if event.widget is self.gallery_frame:
            self.gallery_canvas.configure(
                scrollregion=self.gallery_canvas.bbox("all")
            )

    def _show_movies(self, movies, heading, status):
        self.current_movies = movies
        self.results_title.configure(text=heading)
        self.status_var.set(status)
        self._filter_genre()

    def _on_genre_selected(self, event):
        if event.widget is not None:
            self._filter_genre()

    def _filter_genre(self):
        genre = self.genre_var.get()
        filtered = [
            movie
            for movie in self.current_movies
            if genre == "All genres" or genre in movie[1].split("|")
        ]

        for child in self.gallery_frame.winfo_children():
            child.destroy()

        if not filtered:
            tk.Label(
                self.gallery_frame,
                text="No movies in this genre for the current results.",
                bg="#0b1020",
                fg="#aab6cc",
                font=("Segoe UI", 11),
            ).pack(anchor="w", padx=8, pady=20)
            return

        for index, (title, genres, rating, match) in enumerate(filtered):
            card = tk.Frame(
                self.gallery_frame,
                bg="#151e32",
                width=205,
                height=320,
                padx=12,
                pady=12,
            )
            card.grid(
                row=index // 3,
                column=index % 3,
                sticky="nsew",
                padx=(0, 12),
                pady=(0, 12),
            )
            card.grid_propagate(False)
            poster = tk.Canvas(
                card,
                width=179,
                height=150,
                bg="#202b47",
                highlightthickness=0,
            )
            poster.pack(fill="x")
            colors = ("#55427f", "#2d6271", "#864e62", "#465f42", "#76582f")
            color = colors[sum(ord(char) for char in title) % len(colors)]
            poster.create_rectangle(0, 0, 179, 150, fill=color, outline="")
            poster.create_oval(112, -35, 205, 60, fill="#ffffff", outline="", stipple="gray50")
            poster.create_rectangle(0, 110, 179, 150, fill="#101525", outline="")
            poster.create_text(
                12,
                118,
                text=title[:30],
                anchor="w",
                width=155,
                fill="#ffffff",
                font=("Segoe UI", 10, "bold"),
            )

            tk.Label(
                card,
                text=title,
                bg="#151e32",
                fg="#f3f5fb",
                font=("Segoe UI", 10, "bold"),
                anchor="w",
                justify="left",
                wraplength=179,
            ).pack(fill="x", pady=(8, 3))
            genre_text = genres.replace("|", " · ") or "Genre not listed"
            tk.Label(
                card,
                text=genre_text,
                bg="#151e32",
                fg="#9eacc4",
                font=("Segoe UI", 8),
                anchor="w",
                wraplength=179,
                justify="left",
            ).pack(fill="x")

            if match is not None:
                detail = f"✨  {match:.0%} match"
            elif rating:
                detail = f"★  {rating:.1f}  ·  MovieLens"
            else:
                detail = "MovieLens pick"
            tk.Label(
                card,
                text=detail,
                bg="#151e32",
                fg="#b9aaff",
                font=("Segoe UI", 9, "bold"),
                anchor="w",
            ).pack(fill="x", pady=(5, 2))
            tk.Button(
                card,
                text="Find on YouTube  ↗",
                command=lambda movie_title=title: self._open_movie(movie_title),
                bg="#263451",
                fg="#e8edf7",
                activebackground="#36466a",
                activeforeground="#ffffff",
                relief="flat",
                cursor="hand2",
                font=("Segoe UI", 8, "bold"),
                pady=5,
            ).pack(fill="x", pady=(5, 0))

        for column in range(3):
            self.gallery_frame.grid_columnconfigure(column, weight=1)

    def _open_movie(self, title):
        url = "https://www.youtube.com/results?search_query=" + quote_plus(
            f"{title} official"
        )
        try:
            opened = webbrowser.open(url)
        except webbrowser.Error as error:
            messagebox.showerror(
                "Could not open YouTube",
                f"Your browser could not be opened:\n{error}",
                parent=self.root,
            )
            return

        if not opened:
            messagebox.showerror(
                "Could not open YouTube",
                "No browser could be opened. Please check your default browser.",
                parent=self.root,
            )

    @staticmethod
    def _stat_card(parent, value, label):
        card = tk.Frame(parent, bg="#151e32", padx=18, pady=14)
        tk.Label(
            card,
            text=value,
            bg="#151e32",
            fg="#f3f5fb",
            font=("Segoe UI", 19, "bold"),
        ).pack(anchor="w")
        tk.Label(
            card,
            text=label,
            bg="#151e32",
            fg="#8998b2",
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", pady=(3, 0))
        return card

    def _search(self, _event=None):
        if _event is not None:
            self.search_entry.focus_set()

        movie_name = self.search_var.get().strip()
        if not movie_name:
            self.status_var.set("Type a movie title first.")
            self.search_entry.focus_set()
            return

        actual_title, recommendations = find_recommendations(
            movie_name,
            int(self.number_var.get()),
            self.movies,
            self.similarity_matrix,
        )

        if actual_title is None:
            self.current_movies = []
            self.results_title.configure(text="No matching movies")
            self._filter_genre()
            self.status_var.set(f'No movies found for "{movie_name}". Try another title.')
            return

        self._show_movies(
            [
                (
                    title,
                    genres,
                    float(
                        self.movies.loc[
                            self.movies["title"].eq(title), "average_rating"
                        ].iloc[0]
                    ),
                    score,
                )
                for title, genres, score in recommendations
            ],
            f"Because you searched: {actual_title}",
            f"{len(recommendations)} recommendations · click a card button to find it online.",
        )


def main():
    root = tk.Tk()
    root.withdraw()

    try:
        movies, ratings, similarity_matrix = load_recommender_data()
    except FileNotFoundError as error:
        messagebox.showerror("Movie data not found", str(error), parent=root)
        root.destroy()
        return

    MovieRecommenderApp(root, movies, ratings, similarity_matrix)
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
