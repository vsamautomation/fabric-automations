-- T-SQL CREATE TABLE scripts for Power BI dataset migration
-- Dataset: COVID Bakeoff PBIR
-- Workspace: Auto_DP
-- Generated for SQL Server 2019+

-- This script creates tables based on the Power BI data model,
-- designed for dimensional modeling principles.
-- Foreign Key constraints, Primary Key constraints, and Indexes are intentionally omitted
-- as per the requirements for initial table creation.

----------------------------------------------------------------------------------------------------
-- Table: States
-- Description: Represents geographical states, likely within the US, providing demographic and environmental data.
-- Usage in Power BI: 0 measures, 2 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [States] (
    [Average Temperature ] FLOAT, -- The average temperature recorded for the state.
    [Flag] VARCHAR(255),         -- A flag or indicator associated with the state, possibly for categorization or status.
    [Population] FLOAT,          -- The total population of the state.
    [State] VARCHAR(255)         -- The name of the state.
);

----------------------------------------------------------------------------------------------------
-- Table: OWID COVID data
-- Description: Contains COVID-19 related data from Our World in Data, focusing on new cases by date and country.
-- Usage in Power BI: 10 measures, 2 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [OWID COVID data] (
    [date] DATETIME2(6), -- The specific date for which the COVID data is recorded. (FK to Dates.Date)
    [iso_code] VARCHAR(255), -- The ISO 3166-1 alpha-3 code for the country. (FK to Countries.ISO)
    [New cases] BIGINT   -- The number of new COVID-19 cases reported on this date.
);

----------------------------------------------------------------------------------------------------
-- Table: CGRT Mandates
-- Description: Stores information about government response tracker (CGRT) mandates, likely related to COVID-19 policies.
-- Usage in Power BI: 0 measures, 1 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [CGRT Mandates] (
    [CountryName] VARCHAR(255) -- The name of the country to which the mandate applies. (FK to Countries.Country)
);

----------------------------------------------------------------------------------------------------
-- Table: Cases per US State
-- Description: Detailed COVID-19 case and vaccination data specifically for US states.
-- Usage in Power BI: 14 measures, 2 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [Cases per US State] (
    [Date] DATETIME2(6), -- The specific date for which the case and vaccination data is recorded. (FK to Dates.Date)
    [Incremental cases] BIGINT, -- The number of new COVID-19 cases reported incrementally for the state on this date.
    [People fully vaccinated per hundred] FLOAT, -- The percentage of people fully vaccinated per hundred residents in the state.
    [State] VARCHAR(255), -- The name of the US state. (FK to States.State)
    [people_vaccinated_per_hundred] FLOAT, -- The percentage of people who have received at least one vaccine dose per hundred residents.
    [total_distributed] BIGINT   -- The total number of vaccine doses distributed to the state.
);

----------------------------------------------------------------------------------------------------
-- Table: Cases per country
-- Description: Provides COVID-19 incremental case data aggregated by country.
-- Usage in Power BI: 0 measures, 2 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [Cases per country] (
    [Country] VARCHAR(255), -- The name of the country. (FK to Countries.Country)
    [Date] DATETIME2(6),    -- The specific date for which the case data is recorded. (FK to Dates.Date)
    [IncrementalCases] BIGINT -- The number of new COVID-19 cases reported incrementally for the country on this date.
);

----------------------------------------------------------------------------------------------------
-- Table: Govt Measures
-- Description: Records government measures or interventions, likely related to COVID-19, with implementation and entry dates.
-- Usage in Power BI: 1 measures, 3 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [Govt Measures] (
    [Date implemented] DATETIME2(6), -- The date when a specific government measure was implemented. (FK to Dates.Date)
    [Entry date] DATETIME2(6),       -- The date when this record of the measure was entered or became effective. (FK to LocalDateTable_88205f4b-f7a1-45b0-926f-f9aeda622848.Date)
    [ISO] VARCHAR(255)               -- The ISO 3166-1 alpha-3 code for the country where the measure was implemented. (FK to Countries.ISO)
);

----------------------------------------------------------------------------------------------------
-- Table: Days with restrictions grouped
-- Description: Categorizes and groups days based on various types of COVID-19 restrictions in place for different countries.
-- Usage in Power BI: 0 measures, 1 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [Days with restrictions grouped] (
    [Cancelling public events] VARCHAR(255), -- Indicates the level or status of restrictions on public events.
    [CountryCode] VARCHAR(255),              -- The ISO 3166-1 alpha-3 code for the country. (FK to Countries.ISO)
    [Domestic travel restrictions] VARCHAR(255), -- Indicates the level or status of domestic travel restrictions.
    [Face coverings required] VARCHAR(255),  -- Indicates if face coverings are required and to what extent.
    [International travel controls] VARCHAR(255), -- Indicates the level or status of international travel controls.
    [Public transport closures] VARCHAR(255), -- Indicates the level or status of public transport closures.
    [Restrictions on gathering] VARCHAR(255), -- Indicates the level or status of restrictions on public gatherings.
    [School closures] VARCHAR(255),          -- Indicates the level or status of school closures.
    [Stay at home requirements] VARCHAR(255), -- Indicates the level or status of stay-at-home requirements.
    [Workplace closures] VARCHAR(255)        -- Indicates the level or status of workplace closures.
);

----------------------------------------------------------------------------------------------------
-- Table: GDP History
-- Description: Contains historical Gross Domestic Product (GDP) data, including percentage change, for various countries.
-- Usage in Power BI: 0 measures, 1 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [GDP History] (
    [% change] FLOAT,        -- The percentage change in GDP for the given year.
    [ISO] VARCHAR(255),      -- The ISO 3166-1 alpha-3 code for the country. (FK to Countries.ISO)
    [Year] BIGINT            -- The year for which the GDP data is recorded.
);

----------------------------------------------------------------------------------------------------
-- Table: DateTableTemplate_3fa67ac2-0afb-4cc0-9c50-279baae0411c
-- Description: A standard date dimension table template from Power BI, often used for time intelligence.
-- Usage in Power BI: 0 measures, 0 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [DateTableTemplate_3fa67ac2-0afb-4cc0-9c50-279baae0411c] (
    -- This table is a standard Power BI date table template.
    -- Specific date columns (e.g., Date, Year, Month, Day, etc.) would typically be added here.
    -- No columns were specified in the input for this table, so it's created as an empty structure.
    -- In a full dimensional model, this would contain a comprehensive set of date attributes.
);

----------------------------------------------------------------------------------------------------
-- Table: Countries
-- Description: A dimension table containing information about various countries, serving as a central lookup for country-related data.
-- Usage in Power BI: 2 measures, 7 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [Countries] (
    [Continent] VARCHAR(255), -- The continent where the country is located.
    [Country] VARCHAR(255),   -- The full name of the country.
    [Flag] VARCHAR(255),      -- A flag or indicator associated with the country, possibly for categorization or status.
    [ISO] VARCHAR(255),       -- The ISO 3166-1 alpha-3 code for the country, serving as a unique identifier.
    [Population] BIGINT,      -- The total population of the country.
    [REGION] VARCHAR(255)     -- The geographical region of the country.
);

----------------------------------------------------------------------------------------------------
-- Table: Lats
-- Description: Contains latitude information, likely linked to states.
-- Usage in Power BI: 0 measures, 1 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [Lats] (
    [State] VARCHAR(255) -- The name of the state. (FK to States.State)
    -- This table likely contains latitude data for the specified state, though no specific latitude column was provided.
);

----------------------------------------------------------------------------------------------------
-- Table: Dates
-- Description: A core date dimension table, providing a list of distinct dates for time-based analysis.
-- Usage in Power BI: 0 measures, 4 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [Dates] (
    [Date] DATETIME2(6) -- A specific date, serving as the primary key for this date dimension.
);

----------------------------------------------------------------------------------------------------
-- Table: Days with restrictions
-- Description: Records days on which restrictions were in place for different countries.
-- Usage in Power BI: 0 measures, 1 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [Days with restrictions] (
    [CountryCode] VARCHAR(255) -- The ISO 3166-1 alpha-3 code for the country. (FK to Countries.ISO)
    -- This table likely contains a date column or is implicitly linked to dates to indicate specific days with restrictions.
    -- No specific date column was provided in the input for this table.
);

----------------------------------------------------------------------------------------------------
-- Table: LocalDateTable_88205f4b-f7a1-45b0-926f-f9aeda622848
-- Description: A local date dimension table, often automatically generated by Power BI for specific date contexts.
-- Usage in Power BI: 0 measures, 1 relationships
----------------------------------------------------------------------------------------------------
CREATE TABLE [LocalDateTable_88205f4b-f7a1-45b0-926f-f9aeda622848] (
    [Date] DATETIME2(6) -- A specific date, serving as the primary key for this local date dimension.
);
