-- Optional extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Brands
CREATE TABLE brands (
    brand_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    country VARCHAR(50),
    website VARCHAR(255),
    cruelty_free BOOLEAN DEFAULT FALSE,
    vegan BOOLEAN DEFAULT FALSE
);

-- Product Types
CREATE TABLE product_types (
    type_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT
);

-- Products
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    brand_id INT REFERENCES brands(brand_id) ON DELETE SET NULL,
    type_id INT REFERENCES product_types(type_id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price NUMERIC(10,2),
    currency VARCHAR(10) DEFAULT 'INR',
    rating NUMERIC(3,2),
    review_count INT DEFAULT 0,
    availability BOOLEAN DEFAULT TRUE,
    image_url TEXT,
    product_url TEXT UNIQUE,
    last_scraped TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Hair Types
CREATE TABLE hair_types (
    hair_type_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    sebum_level VARCHAR(20),
    porosity VARCHAR(20),
    density VARCHAR(20)
);

-- Hair Concerns
CREATE TABLE hair_concerns (
    concern_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    cause TEXT,
    severity_scale INT DEFAULT 5
);

-- Ingredients
CREATE TABLE ingredients (
    ingredient_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(50),
    benefits TEXT,
    comedogenic_rating INT,
    is_silicone BOOLEAN DEFAULT FALSE,
    is_sulfate BOOLEAN DEFAULT FALSE,
    is_paraben BOOLEAN DEFAULT FALSE,
    source VARCHAR(50),
    suitable_for_vegan BOOLEAN DEFAULT TRUE
);

-- Product <-> Ingredients (M:N)
CREATE TABLE product_ingredients (
    product_id INT REFERENCES products(product_id) ON DELETE CASCADE,
    ingredient_id INT REFERENCES ingredients(ingredient_id) ON DELETE CASCADE,
    concentration_level VARCHAR(50),
    PRIMARY KEY (product_id, ingredient_id)
);

-- Product <-> Hair Types (M:N)
CREATE TABLE product_hair_types (
    product_id INT REFERENCES products(product_id) ON DELETE CASCADE,
    hair_type_id INT REFERENCES hair_types(hair_type_id) ON DELETE CASCADE,
    suitability_score NUMERIC(3,2) DEFAULT 1.0 CHECK (suitability_score >= 0 AND suitability_score <= 1),
    PRIMARY KEY (product_id, hair_type_id)
);

-- Product <-> Concerns (M:N)
CREATE TABLE product_concerns (
    product_id INT REFERENCES products(product_id) ON DELETE CASCADE,
    concern_id INT REFERENCES hair_concerns(concern_id) ON DELETE CASCADE,
    effectiveness_score NUMERIC(3,2) DEFAULT 1.0 CHECK (effectiveness_score >= 0 AND effectiveness_score <= 1),
    PRIMARY KEY (product_id, concern_id)
);

-- Users
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(255) UNIQUE NOT NULL,
    signup_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User Profiles
CREATE TABLE user_profiles (
    user_id INT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    hair_type_id INT REFERENCES hair_types(hair_type_id),
    primary_concern_id INT REFERENCES hair_concerns(concern_id),
    secondary_concern_id INT REFERENCES hair_concerns(concern_id),
    scalp_sensitivity BOOLEAN DEFAULT FALSE,
    color_treated BOOLEAN DEFAULT FALSE,
    preferred_ingredients TEXT,
    avoid_ingredients TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User Product Ratings
CREATE TABLE user_product_ratings (
    user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
    product_id INT REFERENCES products(product_id) ON DELETE CASCADE,
    rating NUMERIC(2,1) CHECK (rating >= 0 AND rating <= 5),
    review_text TEXT,
    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, product_id)
);
