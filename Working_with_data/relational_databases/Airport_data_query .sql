-- SQLite

-- all city names in the Cities table

SELECT city from Cities;

-- all cities in Ireland in the Cities table

SELECT city FROM Cities WHERE country = 'Ireland';


-- all airport names with their city and country


SELECT a.name AS airport_name, c.city, c.country
FROM Airports a
JOIN Cities c ON a.city_id = c.id;


-- all airports in London, United Kingdom
SELECT a.name AS Airport_in_London
FROM Airports a
JOIN Cities c ON a.city_id = c.id
WHERE c.city LIKE 'London' AND c.country LIKE 'United Kingdom';
