-- Clean SQLite Schema for Receipt Persistence
-- Tables: receipts, line_items
-- Includes indexes and constraints for optimal performance

-- Drop existing tables if they exist (for clean schema)
-- Note: In production, use migration logic instead
DROP TABLE IF EXISTS line_items;
DROP TABLE IF EXISTS receipts;
DROP TABLE IF EXISTS users;

-- Users table (for multi-tenant support)
CREATE TABLE users (
    id TEXT PRIMARY KEY,                    -- UUID as TEXT for SQLite
    email TEXT UNIQUE NOT NULL,             -- User email (unique)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- Receipts table (main receipt data)
CREATE TABLE receipts (
    id TEXT PRIMARY KEY,                    -- UUID as TEXT for SQLite
    user_id TEXT NOT NULL,                  -- Foreign key to users
    image_path TEXT NOT NULL,               -- Path to receipt image
    receipt_hash TEXT UNIQUE NOT NULL,      -- SHA-256 hash for deduplication
    status TEXT NOT NULL DEFAULT 'pending', -- Processing status
    
    -- Receipt metadata
    receipt_date DATE,                      -- Date on receipt
    merchant_name TEXT,                     -- Merchant/store name
    total_amount DECIMAL(10, 2),            -- Total amount
    subtotal DECIMAL(10, 2),                -- Subtotal
    tax_amount DECIMAL(10, 2),              -- Tax amount
    tip_amount DECIMAL(10, 2),              -- Tip amount
    
    -- Processing data
    raw_ocr_text TEXT,                      -- Raw OCR output
    parsed_data TEXT,                       -- JSON parsed data
    ocr_confidence DECIMAL(3, 2),          -- OCR confidence 0.00-1.00
    processing_started_at DATETIME,         -- Processing start time
    processing_completed_at DATETIME,       -- Processing completion time
    processing_error TEXT,                  -- Error message if failed
    retry_count INTEGER DEFAULT 0,         -- Number of retry attempts
    
    -- Cost tracking
    input_tokens INTEGER DEFAULT 0,        -- OpenAI input tokens
    output_tokens INTEGER DEFAULT 0,       -- OpenAI output tokens
    estimated_cost DECIMAL(8, 4) DEFAULT 0.0000, -- Estimated API cost
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraint
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Line items table (receipt line items)
CREATE TABLE line_items (
    id TEXT PRIMARY KEY,                    -- UUID as TEXT for SQLite
    receipt_id TEXT NOT NULL,               -- Foreign key to receipts
    line_number INTEGER NOT NULL,           -- Line number on receipt
    
    -- Item details
    description TEXT NOT NULL,              -- Item description
    quantity DECIMAL(8, 2) DEFAULT 1.00,   -- Quantity
    unit_price DECIMAL(10, 2) NOT NULL,     -- Unit price
    total_price DECIMAL(10, 2) NOT NULL,    -- Total price (quantity * unit_price)
    
    -- Category and tax
    category TEXT,                          -- Item category
    is_taxable BOOLEAN DEFAULT 1,           -- Whether item is taxable
    tax_rate DECIMAL(5, 4) DEFAULT 0.0000,  -- Tax rate (e.g., 0.0825 for 8.25%)
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraint
    FOREIGN KEY (receipt_id) REFERENCES receipts(id) ON DELETE CASCADE
);

-- Indexes for receipts table
-- Core performance indexes
CREATE INDEX idx_receipts_user_id ON receipts(user_id);
CREATE INDEX idx_receipts_status ON receipts(status);
CREATE INDEX idx_receipts_date ON receipts(receipt_date);
CREATE INDEX idx_receipts_created_at ON receipts(created_at);

-- Composite indexes for common query patterns
CREATE INDEX idx_receipts_user_status ON receipts(user_id, status);
CREATE INDEX idx_receipts_user_date ON receipts(user_id, receipt_date);
CREATE INDEX idx_receipts_user_created ON receipts(user_id, created_at);
CREATE INDEX idx_receipts_status_created ON receipts(status, created_at);

-- Unique constraint on receipt_hash (already enforced by column UNIQUE)
-- Additional unique constraint for user-scoped deduplication
CREATE UNIQUE INDEX idx_receipts_user_hash ON receipts(user_id, receipt_hash);

-- Search and analytics indexes
CREATE INDEX idx_receipts_merchant_name ON receipts(merchant_name);
CREATE INDEX idx_receipts_total_amount ON receipts(total_amount);
CREATE INDEX idx_receipts_user_merchant ON receipts(user_id, merchant_name);
CREATE INDEX idx_receipts_user_amount ON receipts(user_id, total_amount);

-- Processing and cost indexes
CREATE INDEX idx_receipts_processing_started ON receipts(processing_started_at);
CREATE INDEX idx_receipts_retry_count ON receipts(retry_count);
CREATE INDEX idx_receipts_estimated_cost ON receipts(estimated_cost);
CREATE INDEX idx_receipts_user_cost ON receipts(user_id, estimated_cost);

-- Indexes for line_items table
-- Core performance indexes
CREATE INDEX idx_line_items_receipt_id ON line_items(receipt_id);
CREATE INDEX idx_line_items_category ON line_items(category);
CREATE INDEX idx_line_items_description ON line_items(description);

-- Composite indexes for common query patterns
CREATE INDEX idx_line_items_receipt_category ON line_items(receipt_id, category);
CREATE INDEX idx_line_items_receipt_description ON line_items(receipt_id, description);
CREATE INDEX idx_line_items_category_price ON line_items(category, unit_price);
CREATE INDEX idx_line_items_receipt_category_price ON line_items(receipt_id, category, unit_price);

-- Price and analytics indexes
CREATE INDEX idx_line_items_unit_price ON line_items(unit_price);
CREATE INDEX idx_line_items_total_price ON line_items(total_price);
CREATE INDEX idx_line_items_quantity ON line_items(quantity);

-- Tax and business logic indexes
CREATE INDEX idx_line_items_is_taxable ON line_items(is_taxable);
CREATE INDEX idx_line_items_taxable_category ON line_items(is_taxable, category);
CREATE INDEX idx_line_items_tax_rate ON line_items(tax_rate);

-- Indexes for users table
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_is_active ON users(is_active);
CREATE INDEX idx_users_created_at ON users(created_at);

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_receipts_updated_at 
    AFTER UPDATE ON receipts
    FOR EACH ROW
BEGIN
    UPDATE receipts SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_line_items_updated_at 
    AFTER UPDATE ON line_items
    FOR EACH ROW
BEGIN
    UPDATE line_items SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_users_updated_at 
    AFTER UPDATE ON users
    FOR EACH ROW
BEGIN
    UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
