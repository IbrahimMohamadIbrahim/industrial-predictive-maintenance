-- Analytical SQL for the AI4I 2020 Predictive Maintenance dataset.
-- These are descriptive analytics only; they do not implement a risk engine.
-- Run after: python scripts/build_database.py

-- Q1: How large is the dataset and what is the overall historical failure rate?
-- Useful as a basic data-volume and class-imbalance sanity check.
SELECT
    COUNT(*) AS observations,
    SUM(machine_failure) AS failures,
    ROUND(100.0 * AVG(CAST(machine_failure AS REAL)), 3) AS failure_rate_pct
FROM failures;

-- Q2: Which product quality type has the highest historical failure rate? (JOIN/GROUP BY)
-- Useful for comparing failure prevalence across the documented L/M/H product variants.
SELECT
    p.product_type,
    COUNT(*) AS observations,
    SUM(f.machine_failure) AS failures,
    ROUND(100.0 * AVG(CAST(f.machine_failure AS REAL)), 3) AS failure_rate_pct
FROM products AS p
JOIN observations AS o ON o.product_id = p.product_id
JOIN failures AS f ON f.udi = o.udi
GROUP BY p.product_type
ORDER BY failure_rate_pct DESC;

-- Q3: Which published failure-mode labels occur most often?
-- Useful for understanding the imbalance among failure modes without treating them as predictors.
SELECT failure_mode, failure_count
FROM (
    SELECT 'TWF' AS failure_mode, SUM(twf) AS failure_count FROM failures
    UNION ALL SELECT 'HDF', SUM(hdf) FROM failures
    UNION ALL SELECT 'PWF', SUM(pwf) FROM failures
    UNION ALL SELECT 'OSF', SUM(osf) FROM failures
    UNION ALL SELECT 'RNF', SUM(rnf) FROM failures
)
ORDER BY failure_count DESC, failure_mode;

-- Q4: What are the average operating measurements by product type? (JOIN/Aggregates)
-- Useful for checking whether operating conditions differ across L/M/H product variants.
SELECT
    p.product_type,
    ROUND(AVG(o.air_temperature_k), 3) AS avg_air_temperature_k,
    ROUND(AVG(o.process_temperature_k), 3) AS avg_process_temperature_k,
    ROUND(AVG(o.rotational_speed_rpm), 3) AS avg_rotational_speed_rpm,
    ROUND(AVG(o.torque_nm), 3) AS avg_torque_nm,
    ROUND(AVG(o.tool_wear_min), 3) AS avg_tool_wear_min
FROM products AS p
JOIN observations AS o ON o.product_id = p.product_id
GROUP BY p.product_type
ORDER BY p.product_type;

-- Q5: How do average operating measurements differ between failed and non-failed observations? (JOIN)
-- Useful for an initial target-focused comparison before formal EDA/modeling.
SELECT
    f.machine_failure,
    COUNT(*) AS observations,
    ROUND(AVG(o.air_temperature_k), 3) AS avg_air_temperature_k,
    ROUND(AVG(o.process_temperature_k), 3) AS avg_process_temperature_k,
    ROUND(AVG(o.rotational_speed_rpm), 3) AS avg_rotational_speed_rpm,
    ROUND(AVG(o.torque_nm), 3) AS avg_torque_nm,
    ROUND(AVG(o.tool_wear_min), 3) AS avg_tool_wear_min
FROM observations AS o
JOIN failures AS f ON f.udi = o.udi
GROUP BY f.machine_failure
ORDER BY f.machine_failure;

-- Q6: Which failed observations have the greatest tool wear? (WHERE/ORDER BY)
-- Useful for inspecting high-wear failure examples without claiming causal relationships.
SELECT
    o.udi,
    o.product_id,
    p.product_type,
    o.tool_wear_min,
    o.torque_nm,
    o.rotational_speed_rpm
FROM observations AS o
JOIN products AS p ON p.product_id = o.product_id
JOIN failures AS f ON f.udi = o.udi
WHERE f.machine_failure = 1
ORDER BY o.tool_wear_min DESC, o.udi
LIMIT 20;

-- Q7: How does failure rate vary across tool-wear bands? (CTE)
-- Useful for a transparent descriptive view of failure prevalence as tool wear increases.
WITH wear_bands AS (
    SELECT
        udi,
        CASE
            WHEN tool_wear_min < 50 THEN '0-49'
            WHEN tool_wear_min < 100 THEN '50-99'
            WHEN tool_wear_min < 150 THEN '100-149'
            WHEN tool_wear_min < 200 THEN '150-199'
            ELSE '200+'
        END AS wear_band,
        CASE
            WHEN tool_wear_min < 50 THEN 1
            WHEN tool_wear_min < 100 THEN 2
            WHEN tool_wear_min < 150 THEN 3
            WHEN tool_wear_min < 200 THEN 4
            ELSE 5
        END AS band_order
    FROM observations
)
SELECT
    w.wear_band,
    COUNT(*) AS observations,
    SUM(f.machine_failure) AS failures,
    ROUND(100.0 * AVG(CAST(f.machine_failure AS REAL)), 3) AS failure_rate_pct
FROM wear_bands AS w
JOIN failures AS f ON f.udi = w.udi
GROUP BY w.wear_band, w.band_order
ORDER BY w.band_order;

-- Q8: Which torque/speed operating bands have the highest historical failure rates? (CTE/GROUP BY)
-- Useful for identifying descriptive operating regimes for deeper EDA; this is not a predictive risk score.
WITH operating_bands AS (
    SELECT
        udi,
        CASE
            WHEN torque_nm < 30 THEN 'low_torque'
            WHEN torque_nm < 50 THEN 'mid_torque'
            ELSE 'high_torque'
        END AS torque_band,
        CASE
            WHEN rotational_speed_rpm < 1400 THEN 'low_speed'
            WHEN rotational_speed_rpm < 1800 THEN 'mid_speed'
            ELSE 'high_speed'
        END AS speed_band
    FROM observations
)
SELECT
    b.torque_band,
    b.speed_band,
    COUNT(*) AS observations,
    SUM(f.machine_failure) AS failures,
    ROUND(100.0 * AVG(CAST(f.machine_failure AS REAL)), 3) AS failure_rate_pct
FROM operating_bands AS b
JOIN failures AS f ON f.udi = b.udi
GROUP BY b.torque_band, b.speed_band
HAVING COUNT(*) >= 20
ORDER BY failure_rate_pct DESC, observations DESC;

-- Q9: Which observations have more than one failure-mode flag set?
-- Useful for examining multi-mode records and label complexity in the published source.
SELECT
    o.udi,
    o.product_id,
    (f.twf + f.hdf + f.pwf + f.osf + f.rnf) AS failure_mode_count,
    f.machine_failure,
    f.twf,
    f.hdf,
    f.pwf,
    f.osf,
    f.rnf
FROM observations AS o
JOIN failures AS f ON f.udi = o.udi
WHERE (f.twf + f.hdf + f.pwf + f.osf + f.rnf) > 1
ORDER BY failure_mode_count DESC, o.udi;

-- Q10: Does failure prevalence change across the ordered UDI sequence? (CTE)
-- UDI is treated only as source order, not as a real timestamp; this can flag sequence-level shifts for investigation.
WITH sequence_blocks AS (
    SELECT
        udi,
        ((udi - 1) / 1000) + 1 AS udi_block
    FROM observations
)
SELECT
    s.udi_block,
    MIN(s.udi) AS first_udi,
    MAX(s.udi) AS last_udi,
    COUNT(*) AS observations,
    SUM(f.machine_failure) AS failures,
    ROUND(100.0 * AVG(CAST(f.machine_failure AS REAL)), 3) AS failure_rate_pct
FROM sequence_blocks AS s
JOIN failures AS f ON f.udi = s.udi
GROUP BY s.udi_block
ORDER BY s.udi_block;

-- Q11: What is the cumulative failure rate within each product type in source order? (Window Function)
-- Useful for demonstrating a window calculation while preserving the dataset's sequence semantics.
SELECT
    o.udi,
    p.product_type,
    f.machine_failure,
    SUM(f.machine_failure) OVER (
        PARTITION BY p.product_type
        ORDER BY o.udi
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_failures,
    COUNT(*) OVER (
        PARTITION BY p.product_type
        ORDER BY o.udi
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_observations,
    ROUND(
        100.0 * SUM(f.machine_failure) OVER (
            PARTITION BY p.product_type
            ORDER BY o.udi
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) /
        COUNT(*) OVER (
            PARTITION BY p.product_type
            ORDER BY o.udi
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        3
    ) AS cumulative_failure_rate_pct
FROM observations AS o
JOIN products AS p ON p.product_id = o.product_id
JOIN failures AS f ON f.udi = o.udi
ORDER BY o.udi
LIMIT 100;

-- Q12: Rank product types by failure rate and show their share of all failures. (CTE + Window Function)
-- Useful for combining group-level failure statistics with relative ranking/contribution.
WITH type_stats AS (
    SELECT
        p.product_type,
        COUNT(*) AS observations,
        SUM(f.machine_failure) AS failures,
        AVG(CAST(f.machine_failure AS REAL)) AS failure_rate
    FROM products AS p
    JOIN observations AS o ON o.product_id = p.product_id
    JOIN failures AS f ON f.udi = o.udi
    GROUP BY p.product_type
)
SELECT
    product_type,
    observations,
    failures,
    ROUND(100.0 * failure_rate, 3) AS failure_rate_pct,
    DENSE_RANK() OVER (ORDER BY failure_rate DESC) AS failure_rate_rank,
    ROUND(100.0 * failures / SUM(failures) OVER (), 3) AS share_of_all_failures_pct
FROM type_stats
ORDER BY failure_rate_rank, product_type;
