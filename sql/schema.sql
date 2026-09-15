-- Relational schema for the UCI AI4I 2020 Predictive Maintenance dataset.
-- One source row represents one process observation identified by UDI.
-- Product ID is unique per observation in the published dataset; it is NOT a
-- persistent physical-machine identifier, so the schema avoids inventing a
-- machine-history concept that the source data does not contain.

DROP VIEW IF EXISTS failure_observation_details;
DROP TABLE IF EXISTS failures;
DROP TABLE IF EXISTS observations;
DROP TABLE IF EXISTS products;

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_type TEXT NOT NULL CHECK (product_type IN ('L', 'M', 'H')),
    CHECK (substr(product_id, 1, 1) = product_type)
);

CREATE TABLE observations (
    udi INTEGER PRIMARY KEY CHECK (udi BETWEEN 1 AND 10000),
    product_id TEXT NOT NULL UNIQUE,
    air_temperature_k REAL NOT NULL CHECK (air_temperature_k > 0),
    process_temperature_k REAL NOT NULL CHECK (process_temperature_k > 0),
    rotational_speed_rpm INTEGER NOT NULL CHECK (rotational_speed_rpm > 0),
    torque_nm REAL NOT NULL CHECK (torque_nm >= 0),
    tool_wear_min INTEGER NOT NULL CHECK (tool_wear_min >= 0),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE failures (
    udi INTEGER PRIMARY KEY,
    machine_failure INTEGER NOT NULL CHECK (machine_failure IN (0, 1)),
    twf INTEGER NOT NULL CHECK (twf IN (0, 1)),
    hdf INTEGER NOT NULL CHECK (hdf IN (0, 1)),
    pwf INTEGER NOT NULL CHECK (pwf IN (0, 1)),
    osf INTEGER NOT NULL CHECK (osf IN (0, 1)),
    rnf INTEGER NOT NULL CHECK (rnf IN (0, 1)),
    FOREIGN KEY (udi) REFERENCES observations(udi)
);

CREATE INDEX idx_products_type ON products(product_type);
CREATE INDEX idx_failures_machine_failure ON failures(machine_failure);
CREATE INDEX idx_observations_tool_wear ON observations(tool_wear_min);

CREATE VIEW failure_observation_details AS
SELECT
    o.udi,
    o.product_id,
    p.product_type,
    o.air_temperature_k,
    o.process_temperature_k,
    o.rotational_speed_rpm,
    o.torque_nm,
    o.tool_wear_min,
    f.machine_failure,
    f.twf,
    f.hdf,
    f.pwf,
    f.osf,
    f.rnf
FROM observations AS o
JOIN products AS p ON p.product_id = o.product_id
JOIN failures AS f ON f.udi = o.udi;
