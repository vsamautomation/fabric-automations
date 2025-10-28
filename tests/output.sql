`CREATED BY GEMINI`
-- T-SQL script to create tables for the New Waziri Dashboard Report Power BI dataset.
-- Workspace: Test Workspace

----------------------------------------------------------------------------------------------------
-- Table: Payments
-- Purpose: Stores payment information.
----------------------------------------------------------------------------------------------------
CREATE TABLE Payments (
    date DATETIME2 NOT NULL,
    customer VARCHAR(255) NOT NULL,
    CONSTRAINT PK_Payments PRIMARY KEY (date, customer) -- Composite key for better uniqueness
);

-- Add foreign key constraint
ALTER TABLE Payments
ADD CONSTRAINT FK_Payments_LocalDateTable_039beab2
FOREIGN KEY (date)
REFERENCES LocalDateTable_039beab2_6ebe_415d_b0cb_70b10b1651f4(Date);

-- Add foreign key constraint
ALTER TABLE Payments
ADD CONSTRAINT FK_Payments_CustomersDB
FOREIGN KEY (customer)
REFERENCES CustomersDB([customer name]);

-- Create index on foreign key column
CREATE INDEX IX_Payments_date ON Payments (date);
CREATE INDEX IX_Payments_customer ON Payments (customer);

----------------------------------------------------------------------------------------------------
-- Table: hpeqhdmy_salesapp vehicles
-- Purpose: Stores vehicle makes.
----------------------------------------------------------------------------------------------------
CREATE TABLE [hpeqhdmy_salesapp vehicles] (
    Vehicle_Make VARCHAR(255) NOT NULL,
    CONSTRAINT PK_hpeqhdmy_salesapp_vehicles PRIMARY KEY (Vehicle_Make)
);

----------------------------------------------------------------------------------------------------
-- Table: ItemsDB
-- Purpose: Stores item details.
----------------------------------------------------------------------------------------------------
CREATE TABLE ItemsDB (
    [part number] VARCHAR(255) NOT NULL,
    [item name] VARCHAR(255) NULL,
    [vehicle make] VARCHAR(255) NULL,
    [image url] VARCHAR(255) NULL,
    CONSTRAINT PK_ItemsDB PRIMARY KEY ([part number])
);

-- Add foreign key constraint
ALTER TABLE ItemsDB
ADD CONSTRAINT FK_ItemsDB_hpeqhdmy_salesapp_vehicles
FOREIGN KEY ([vehicle make])
REFERENCES [hpeqhdmy_salesapp vehicles](Vehicle_Make);

-- Create index on foreign key column
CREATE INDEX IX_ItemsDB_vehicle_make ON ItemsDB ([vehicle make]);

----------------------------------------------------------------------------------------------------
-- Table: LocalDateTable_039beab2-6ebe-415d-b0cb-70b10b1651f4
-- Purpose: Date table.
----------------------------------------------------------------------------------------------------
CREATE TABLE LocalDateTable_039beab2_6ebe_415d_b0cb_70b10b1651f4 (
    Date DATETIME2 NOT NULL,
    CONSTRAINT PK_LocalDateTable_039beab2 PRIMARY KEY (Date)
);

----------------------------------------------------------------------------------------------------
-- Table: waz_stock_movement
-- Purpose: Stores stock movement data.
----------------------------------------------------------------------------------------------------
CREATE TABLE waz_stock_movement (
    date DATETIME2 NOT NULL,
    CONSTRAINT PK_waz_stock_movement PRIMARY KEY (date)
);

-- Add foreign key constraint
ALTER TABLE waz_stock_movement
ADD CONSTRAINT FK_waz_stock_movement_LocalDateTable_6342b19c
FOREIGN KEY (date)
REFERENCES LocalDateTable_6342b19c_5456_4a79_98a6_fd9a5025ceda(Date);

-- Create index on foreign key column
CREATE INDEX IX_waz_stock_movement_date ON waz_stock_movement (date);

----------------------------------------------------------------------------------------------------
-- Table: _Subs
-- Purpose: Placeholder table for measures.  No columns.
----------------------------------------------------------------------------------------------------
CREATE TABLE _Subs (
    -- No columns
);

----------------------------------------------------------------------------------------------------
-- Table: CustomersDB
-- Purpose: Stores customer information.
----------------------------------------------------------------------------------------------------
CREATE TABLE CustomersDB (
    [customer name] VARCHAR(255) NOT NULL,
    CONSTRAINT PK_CustomersDB PRIMARY KEY ([customer name])
);

----------------------------------------------------------------------------------------------------
-- Table: _Totals
-- Purpose: Placeholder table for measures. No columns.
----------------------------------------------------------------------------------------------------
CREATE TABLE _Totals (
    -- No columns
);

----------------------------------------------------------------------------------------------------
-- Table: Calendar
-- Purpose: Stores calendar data.
----------------------------------------------------------------------------------------------------
CREATE TABLE Calendar (
    Date DATETIME2 NOT NULL,
    Year BIGINT NULL,
    CurrYearOffset BIGINT NULL,
    Quarter VARCHAR(255) NULL,
    Month BIGINT NULL,
    CurrMonthOffset BIGINT NULL,
    [Month Short] VARCHAR(255) NULL,
    [Day of Month] BIGINT NULL,
    CurrWeekOffset BIGINT NULL,
    [Day of Week Name] VARCHAR(255) NULL,
    CONSTRAINT PK_Calendar PRIMARY KEY (Date)
);

----------------------------------------------------------------------------------------------------
-- Table: Orders
-- Purpose: Stores order information.
----------------------------------------------------------------------------------------------------
CREATE TABLE Orders (
    date DATETIME2 NOT NULL,
    customer VARCHAR(255) NOT NULL,
    order_id BIGINT NOT NULL,
    CONSTRAINT PK_Orders PRIMARY KEY (order_id)
);

-- Add foreign key constraint
ALTER TABLE Orders
ADD CONSTRAINT FK_Orders_Calendar
FOREIGN KEY (date)
REFERENCES Calendar(Date);

-- Add foreign key constraint
ALTER TABLE Orders
ADD CONSTRAINT FK_Orders_CustomersDB
FOREIGN KEY (customer)
REFERENCES CustomersDB([customer name]);

-- Create index on foreign key column
CREATE INDEX IX_Orders_date ON Orders (date);
CREATE INDEX IX_Orders_customer ON Orders (customer);

----------------------------------------------------------------------------------------------------
-- Table: LocalDateTable_ee40366b-f8fa-46c1-a0aa-20db619bcf17
-- Purpose: Date table.
----------------------------------------------------------------------------------------------------
CREATE TABLE LocalDateTable_ee40366b_f8fa_46c1_a0aa_20db619bcf17 (
    Date DATETIME2 NOT NULL,
    CONSTRAINT PK_LocalDateTable_ee40366b PRIMARY KEY (Date)
);

----------------------------------------------------------------------------------------------------
-- Table: DateTableTemplate_7b741e0b-d234-4b07-bd1d-3fb38362da96
-- Purpose: Placeholder table. No columns.
----------------------------------------------------------------------------------------------------
CREATE TABLE DateTableTemplate_7b741e0b_d234_4b07_bd1d_3fb38362da96 (
    -- No columns
);

----------------------------------------------------------------------------------------------------
-- Table: LocalDateTable_6342b19c-5456-4a79-98a6-fd9a5025ceda
-- Purpose: Date table.
----------------------------------------------------------------------------------------------------
CREATE TABLE LocalDateTable_6342b19c_5456_4a79_98a6_fd9a5025ceda (
    Date DATETIME2 NOT NULL,
    CONSTRAINT PK_LocalDateTable_6342b19c PRIMARY KEY (Date)
);

----------------------------------------------------------------------------------------------------
-- Table: Sales
-- Purpose: Stores sales transaction data.
----------------------------------------------------------------------------------------------------
CREATE TABLE Sales (
    date DATETIME2 NOT NULL,
    order_id BIGINT NOT NULL,
    customer VARCHAR(255) NOT NULL,
    item VARCHAR(255) NULL,
    part_no VARCHAR(255) NULL,
    price FLOAT NULL,
    profit FLOAT NULL,
    CONSTRAINT PK_Sales PRIMARY KEY (date, order_id, customer) -- Composite key for uniqueness
);

-- Add foreign key constraint
ALTER TABLE Sales
ADD CONSTRAINT FK_Sales_LocalDateTable_ee40366b
FOREIGN KEY (date)
REFERENCES LocalDateTable_ee40366b_f8fa_46c1_a0aa_20db619bcf17(Date);

-- Add foreign key constraint
ALTER TABLE Sales
ADD CONSTRAINT FK_Sales_Orders
FOREIGN KEY (order_id)
REFERENCES Orders(order_id);

-- Add foreign key constraint
ALTER TABLE Sales
ADD CONSTRAINT FK_Sales_CustomersDB
FOREIGN KEY (customer)
REFERENCES CustomersDB([customer name]);

-- Create index on foreign key column
CREATE INDEX IX_Sales_date ON Sales (date);
CREATE INDEX IX_Sales_order_id ON Sales (order_id);
CREATE INDEX IX_Sales_customer ON Sales (customer);
