CREATE DATABASE IF NOT EXISTS steam_games
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE steam_games;

CREATE TABLE IF NOT EXISTS games(
    app_id BIGINT PRIMARY KEY,
    game_name VARCHAR(300) NOT NULL,

    release_date DATE NULL,
    estimated_owners VARCHAR(50) NULL,
    required_age INT NULL,

    price DECIMAL(10,2) NULL,
    discount INT NULL,
    dlc_count INT NULL,

    about_the_game LONGTEXT NULL,
    supported_languages TEXT NULL,
    full_audio_languages TEXT NULL,

    reviews LONGTEXT NULL,
    
    header_image TEXT NULL,
    website TEXT NULL,
    support_url TEXT NULL,
    support_email VARCHAR(320) NULL,

    support_window TINYINT(1) NULL,
    support_mac TINYINT(1) NULL,
    support_linux TINYINT(1) NULL,

    metacritic_score INT NULL,
    metacritic_url TEXT NULL,

    positive INT NULL,
    negative INT NULL,
    achievements INT NULL,
    recommendations INT NULL,

    notes LONGTEXT NULL,

    developers TEXT NULL,
    publishers TEXT NULL,

    categories TEXT NULL,
    genres TEXT NULL,
    tags TEXT NULL,

    screenshots LONGTEXT NULL

);

CREATE INDEX idx_release_date ON games(release_date);
CREATE INDEX idx_price ON games(price);
CREATE INDEX idx_linux ON games(support_linux);
CREATE INDEX idx_publishers ON games(publishers(50));
CREATE INDEX idx_required_age ON games(required_age);