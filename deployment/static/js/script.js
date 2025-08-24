// Movie Recommendation System - Frontend JavaScript
class MovieRecommendationApp {
    constructor() {
        this.selectedMovie = null;
        this.currentRecommendations = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadSystemStats();
        console.log('🎬 Movie Recommendation App initialized');
    }

    setupEventListeners() {
        // Search functionality
        const searchBtn = document.getElementById('search-btn');
        const searchInput = document.getElementById('movie-search');
        
        searchBtn.addEventListener('click', () => this.searchMovies());
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.searchMovies();
            }
        });

        // Auto-search on input (debounced)
        let searchTimeout;
        searchInput.addEventListener('input', () => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                if (searchInput.value.trim().length >= 3) {
                    this.searchMovies();
                }
            }, 500);
        });

        // Generate recommendations button
        const generateBtn = document.getElementById('generate-btn');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.generateRecommendations());
        }

        // Recommendation type change
        const recTypeSelect = document.getElementById('rec-type');
        recTypeSelect.addEventListener('change', () => {
            if (this.selectedMovie) {
                this.generateRecommendations();
            }
        });

        // Number of recommendations change
        const numRecsSelect = document.getElementById('num-recs');
        numRecsSelect.addEventListener('change', () => {
            if (this.selectedMovie) {
                this.generateRecommendations();
            }
        });

        // Suggestion buttons
        document.querySelectorAll('.suggestion-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const movieTitle = btn.dataset.movie;
                searchInput.value = movieTitle;
                this.searchMovies();
            });
        });

        // Modal functionality
        this.setupModalListeners();
    }

    setupModalListeners() {
        const statsBtn = document.getElementById('stats-btn');
        const aboutBtn = document.getElementById('about-btn');

        statsBtn.addEventListener('click', () => {
            this.openModal('stats-modal');
            this.loadSystemStats();
        });

        aboutBtn.addEventListener('click', () => {
            this.openModal('about-modal');
        });

        document.querySelectorAll('.close-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modal = e.target.closest('.modal');
                this.closeModal(modal.id);
            });
        });

        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.closeModal(modal.id);
                }
            });
        });

        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('error-close')) {
                this.hideMessage('error-message');
            }
        });
    }

    async searchMovies() {
        const query = document.getElementById('movie-search').value.trim();
        
        if (!query) {
            this.showError('Please enter a movie title');
            return;
        }

        this.showLoading();

        try {
            const response = await fetch('/search_movie', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: query })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Search failed');

            this.displaySearchResults(data.results || [], data.message);
        } catch (error) {
            console.error('Search error:', error);
            this.showError(`Search failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    displaySearchResults(movies, message = null) {
        const resultsContainer = document.getElementById('search-results');
        const resultsList = document.getElementById('search-results-list');

        if (movies.length === 0) {
            resultsContainer.classList.add('hidden');
            this.showError(message || 'No movies found');
            return;
        }

        resultsList.innerHTML = '';

        movies.forEach(movie => {
            const genres = Array.isArray(movie.genres)
                ? movie.genres.join(', ')
                : (movie.genres ? movie.genres.replace(/\|/g, ', ') : 'N/A');

            const movieElement = document.createElement('div');
            movieElement.className = 'search-result-item';
            movieElement.innerHTML = `
                <div class="search-result-title">${movie.title}</div>
                <div class="search-result-genres">${genres}</div>
            `;

            movieElement.addEventListener('click', () => {
                this.selectMovie(movie);
                resultsContainer.classList.add('hidden');
            });

            resultsList.appendChild(movieElement);
        });

        resultsContainer.classList.remove('hidden');
        resultsContainer.classList.add('fade-in');
    }

    selectMovie(movie) {
        this.selectedMovie = movie;
        document.getElementById('movie-search').value = movie.title;
        this.displaySelectedMovie(movie);
        document.getElementById('selected-movie').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    displaySelectedMovie(movie) {
        const container = document.getElementById('selected-movie');
        const detailsContainer = document.getElementById('movie-details');

        const genres = Array.isArray(movie.genres)
            ? movie.genres
            : (movie.genres ? movie.genres.split('|') : []);

        detailsContainer.innerHTML = `
            <div class="movie-title">${movie.title}</div>
            <div class="movie-genres">
                ${genres.map(genre => `<span class="genre-badge">${genre}</span>`).join('')}
            </div>
            <div class="movie-rating">Movie ID: ${movie.movieId}</div>
        `;

        container.classList.remove('hidden');
        container.classList.add('slide-up');
    }

    async generateRecommendations() {
        if (!this.selectedMovie) {
            this.showError('Please select a movie first');
            return;
        }

        const recType = document.getElementById('rec-type').value;
        const numRecs = parseInt(document.getElementById('num-recs').value);

        this.showLoading();

        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    movie_title: this.selectedMovie.title,
                    type: recType,
                    num_recommendations: numRecs
                })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Recommendation failed');

            this.displayRecommendations(data);
            this.showSuccess(`Found ${data.recommendations.length} recommendations!`);
        } catch (error) {
            console.error('Recommendation error:', error);
            this.showError(`Recommendation failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    displayRecommendations(data) {
        const container = document.getElementById('recommendations');
        const grid = document.getElementById('recommendations-grid');
        const title = document.getElementById('recommendations-title');
        const count = document.getElementById('rec-count');
        const typeDisplay = document.getElementById('rec-type-display');

        title.textContent = `Recommendations for "${data.movie_details.title}"`;
        count.textContent = `${data.count} movies found`;
        typeDisplay.textContent = this.getRecTypeLabel(data.type);

        grid.innerHTML = '';

        if (data.recommendations.length === 0) {
            grid.innerHTML = `
                <div style="grid-column: 1 / -1; text-align: center; padding: 2rem; color: var(--text-secondary);">
                    <h3>No recommendations found</h3>
                    <p>Try a different recommendation type or movie.</p>
                </div>
            `;
        } else {
            data.recommendations.forEach((rec, index) => {
                const recElement = this.createRecommendationCard(rec, index);
                grid.appendChild(recElement);
            });
        }

        container.classList.remove('hidden');
        container.classList.add('slide-up');
        setTimeout(() => container.scrollIntoView({ behavior: 'smooth', block: 'start' }), 300);
    }

    createRecommendationCard(rec, index) {
        const card = document.createElement('div');
        card.className = 'recommendation-card';
        card.style.animationDelay = `${index * 0.1}s`;

        const genres = Array.isArray(rec.genres)
            ? rec.genres
            : (rec.genres ? rec.genres.split('|') : []);

        let scoreHtml = '';
        if (rec.score !== null && rec.score !== undefined) {
            if (rec.content_score !== undefined && rec.collab_score !== undefined) {
                scoreHtml = `
                    <div class="recommendation-score">
                        <span class="score-value">${rec.score.toFixed(3)}</span>
                        <span>Overall</span>
                    </div>
                    <div class="recommendation-score" style="font-size: 0.8rem; margin-top: 0.25rem;">
                        Content: ${rec.content_score.toFixed(3)} | Collab: ${rec.collab_score.toFixed(3)}
                    </div>
                `;
            } else {
                scoreHtml = `
                    <div class="recommendation-score">
                        <span class="score-value">${rec.score.toFixed(3)}</span>
                        <span>Score</span>
                    </div>
                `;
            }
        }

        card.innerHTML = `
            <div class="recommendation-title">${rec.title}</div>
            <div class="recommendation-genres">
                ${genres.map(genre => `<span class="genre-badge">${genre}</span>`).join('')}
            </div>
            ${scoreHtml}
        `;

        card.addEventListener('click', () => this.showMovieDetails(rec));
        return card;
    }

    showMovieDetails(movie) {
        console.log('Movie details:', movie);
        this.showSuccess(`Selected: ${movie.title}`);
    }

    getRecTypeLabel(type) {
        const labels = { content: 'Content-Based', collaborative: 'Collaborative', hybrid: 'Hybrid' };
        return labels[type] || type;
    }

    async loadSystemStats() {
        try {
            const response = await fetch('/stats');
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to load stats');
            this.displayStats(data);
        } catch (error) {
            console.error('Stats error:', error);
            this.showError(`Failed to load stats: ${error.message}`);
        }
    }

    displayStats(stats) {
        const container = document.getElementById('stats-content');
        const safeNumber = (value) => (value !== undefined && value !== null) ? value.toLocaleString() : "N/A";

        container.innerHTML = `
            <div class="stat-item">
                <span class="stat-value">${safeNumber(stats.total_movies)}</span>
                <div class="stat-label">Total Movies</div>
            </div>
            <div class="stat-item">
                <span class="stat-value">${safeNumber(stats.total_ratings)}</span>
                <div class="stat-label">Total Ratings</div>
            </div>
            <div class="stat-item">
                <span class="stat-value">${safeNumber(stats.total_users)}</span>
                <div class="stat-label">Total Users</div>
            </div>
            <div class="stat-item">
                <span class="stat-value">${stats.avg_rating ?? "N/A"}</span>
                <div class="stat-label">Average Rating</div>
            </div>
            <div class="stat-item">
                <span class="stat-value">${stats.total_genres ?? "N/A"}</span>
                <div class="stat-label">Total Genres</div>
            </div>
        `;
    }

    // Utility methods
    openModal(modalId) {
        const modal = document.getElementById(modalId);
        modal.classList.remove('hidden');
        modal.classList.add('fade-in');
        document.body.style.overflow = 'hidden';
    }

    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        modal.classList.add('hidden');
        document.body.style.overflow = 'auto';
    }

    showLoading() {
        document.getElementById('loading-screen').classList.remove('hidden');
    }

    hideLoading() {
        document.getElementById('loading-screen').classList.add('hidden');
    }

    showError(message) {
        const errorDiv = document.getElementById('error-message');
        const errorText = document.getElementById('error-text');
        errorText.textContent = message;
        errorDiv.classList.remove('hidden');
        setTimeout(() => this.hideMessage('error-message'), 5000);
    }

    showSuccess(message) {
        const successDiv = document.getElementById('success-message');
        const successText = document.getElementById('success-text');
        successText.textContent = message;
        successDiv.classList.remove('hidden');
        setTimeout(() => this.hideMessage('success-message'), 3000);
    }

    hideMessage(messageId) {
        document.getElementById(messageId).classList.add('hidden');
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.movieApp = new MovieRecommendationApp();
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal:not(.hidden)').forEach(modal => {
            window.movieApp.closeModal(modal.id);
        });
    }
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        document.getElementById('movie-search').focus();
    }
});

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    .recommendation-card {
        animation: slideInUp 0.5s ease-out forwards;
        opacity: 0;
        transform: translateY(30px);
    }
    @keyframes slideInUp {
        to { opacity: 1; transform: translateY(0); }
    }
`;
document.head.appendChild(style);