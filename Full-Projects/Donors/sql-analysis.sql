SHOW VARIABLES LIKE 'local_infile';
SET GLOBAL local_infile = 1;

CREATE DATABASE IF NOT EXISTS donations_project;
USE donations_project;
DROP TABLE IF EXISTS base;

CREATE TABLE base (
    FanID BIGINT,
    AccountID BIGINT NULL,
    CRMID BIGINT NULL,
    state VARCHAR(10),
    county VARCHAR(100),
    Region VARCHAR(100),
    AccountType VARCHAR(50),
    YearsOfDonating INT,
    CurrentYearMembershipLevel VARCHAR(100),
    MostRecentMembershipLevel VARCHAR(100),
    MostRecentMembershipYear INT,
    HighestMembershipLevel VARCHAR(100),
    LowestMembershipLevel VARCHAR(100),
    PriorYearMembershipLevel VARCHAR(100),
    TotalLifetimeMembershipDonationDollars DECIMAL(14,2),
    LifetimeGivingRank INT,
    IsAlumni CHAR(1),
    IsSportParticipant CHAR(1),
    PYAnnualFundPledged DECIMAL(14,2),
    PYAnnualFundPaid DECIMAL(14,2),
    CYAnnualFundPledged DECIMAL(14,2),
    CYAnnualFundPaid DECIMAL(14,2),
    CurrentYearMembershipPledged DECIMAL(14,2),
    CurrentYearMembershipPaid DECIMAL(14,2),
    CurrentYearOtherPledged DECIMAL(14,2),
    CurrentYearOtherPaid DECIMAL(14,2)
);

DROP TABLE IF EXISTS ten_year_donation;

CREATE TABLE ten_year_donation (
    FanID BIGINT,
    FundGroupName VARCHAR(150),
    FundName VARCHAR(200),
    FundCode VARCHAR(50),
    MostRecentPaymentPledgeDate DATETIME NULL,
    DatePledged DATETIME NULL,
    DriveYear INT,
    PledgeAmount DECIMAL(14,2),
    PaymentAmount DECIMAL(14,2)
);


DROP TABLE IF EXISTS demographic;

CREATE TABLE demographic (
    FanID BIGINT,
    Age INT
);

DROP TABLE IF EXISTS education;

CREATE TABLE education (
    FanID BIGINT,
    Institution VARCHAR(150),
    EducationStatus VARCHAR(100),
    EducationDate DATETIME NULL,
    Degree VARCHAR(150),
    DegreeType VARCHAR(50),
    College VARCHAR(150),
    Division VARCHAR(150),
    Department VARCHAR(150)
);

DROP TABLE IF EXISTS board_members;

CREATE TABLE board_members (
    FanID BIGINT,
    Board VARCHAR(200),
    RoleName VARCHAR(100),
    StartDate DATETIME NULL,
    EndDate DATETIME NULL,
    CurrentFlag TINYINT
);

DROP TABLE IF EXISTS wealth;

CREATE TABLE wealth (
    FanID BIGINT,
    IWave_Properties DECIMAL(14,2),
    IWave_PropertiesValue DECIMAL(16,2),
    IWave_AnnualCapacity DECIMAL(16,2),
    IWave_AnnualCapacityTargetProduction DECIMAL(16,2)
);

# BASE
LOAD DATA LOCAL INFILE '../base.csv'
INTO TABLE base
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(@FanID, @AccountID, @CRMID, @state, @county, @Region, @AccountType, @YearsOfDonating,
 @CurrentYearMembershipLevel, @MostRecentMembershipLevel, @MostRecentMembershipYear,
 @HighestMembershipLevel, @LowestMembershipLevel, @PriorYearMembershipLevel,
 @TotalLifetimeMembershipDonationDollars, @LifetimeGivingRank, @IsAlumni, @IsSportParticipant,
 @PYAnnualFundPledged, @PYAnnualFundPaid, @CYAnnualFundPledged, @CYAnnualFundPaid,
 @CurrentYearMembershipPledged, @CurrentYearMembershipPaid, @CurrentYearOtherPledged, @CurrentYearOtherPaid)
SET
FanID = NULLIF(@FanID, ''),
AccountID = NULLIF(REPLACE(@AccountID, '.0', ''), ''),
CRMID = NULLIF(REPLACE(@CRMID, '.0', ''), ''),
state = NULLIF(@state, ''),
county = NULLIF(@county, ''),
Region = NULLIF(@Region, ''),
AccountType = NULLIF(@AccountType, ''),
YearsOfDonating = NULLIF(@YearsOfDonating, ''),
CurrentYearMembershipLevel = NULLIF(@CurrentYearMembershipLevel, 'null'),
MostRecentMembershipLevel = NULLIF(@MostRecentMembershipLevel, 'null'),
MostRecentMembershipYear = NULLIF(@MostRecentMembershipYear, ''),
HighestMembershipLevel = NULLIF(@HighestMembershipLevel, 'null'),
LowestMembershipLevel = NULLIF(@LowestMembershipLevel, 'null'),
PriorYearMembershipLevel = NULLIF(@PriorYearMembershipLevel, 'null'),
TotalLifetimeMembershipDonationDollars = NULLIF(@TotalLifetimeMembershipDonationDollars, ''),
LifetimeGivingRank = NULLIF(@LifetimeGivingRank, ''),
IsAlumni = NULLIF(@IsAlumni, ''),
IsSportParticipant = NULLIF(@IsSportParticipant, ''),
PYAnnualFundPledged = NULLIF(@PYAnnualFundPledged, ''),
PYAnnualFundPaid = NULLIF(@PYAnnualFundPaid, ''),
CYAnnualFundPledged = NULLIF(@CYAnnualFundPledged, ''),
CYAnnualFundPaid = NULLIF(@CYAnnualFundPaid, ''),
CurrentYearMembershipPledged = NULLIF(@CurrentYearMembershipPledged, ''),
CurrentYearMembershipPaid = NULLIF(@CurrentYearMembershipPaid, ''),
CurrentYearOtherPledged = NULLIF(@CurrentYearOtherPledged, ''),
CurrentYearOtherPaid = NULLIF(@CurrentYearOtherPaid, '');

# TEN YEAR DONATION
LOAD DATA LOCAL INFILE '../ten_year_donation.csv'
INTO TABLE ten_year_donation
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(@FanID, @FundGroupName, @FundName, @FundCode, @MostRecentPaymentPledgeDate,
 @DatePledged, @DriveYear, @PledgeAmount, @PaymentAmount)
SET
FanID = NULLIF(@FanID, ''),
FundGroupName = NULLIF(@FundGroupName, ''),
FundName = NULLIF(@FundName, ''),
FundCode = NULLIF(@FundCode, ''),
MostRecentPaymentPledgeDate = CASE
    WHEN NULLIF(@MostRecentPaymentPledgeDate, '') IS NULL THEN NULL
    ELSE STR_TO_DATE(@MostRecentPaymentPledgeDate, '%Y-%m-%d %H:%i:%s')
END,
DatePledged = CASE
    WHEN NULLIF(@DatePledged, '') IS NULL THEN NULL
    ELSE STR_TO_DATE(@DatePledged, '%Y-%m-%d %H:%i:%s')
END,
DriveYear = NULLIF(@DriveYear, ''),
PledgeAmount = NULLIF(@PledgeAmount, ''),
PaymentAmount = NULLIF(@PaymentAmount, '');

# DEMOGRAPHIC
LOAD DATA LOCAL INFILE '../demographic.csv'
INTO TABLE demographic
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(@FanID, @Age)
SET
FanID = NULLIF(@FanID, ''),
Age = NULLIF(@Age, '');

# EDUCATION
LOAD DATA LOCAL INFILE '../education.csv'
INTO TABLE education
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(@FanID, @Institution, @EducationStatus, @EducationDate, @Degree,
 @DegreeType, @College, @Division, @Department)
SET
FanID = NULLIF(@FanID, ''),
Institution = NULLIF(@Institution, ''),
EducationStatus = NULLIF(@EducationStatus, ''),
EducationDate = CASE
    WHEN NULLIF(@EducationDate, '') IS NULL THEN NULL
    ELSE STR_TO_DATE(@EducationDate, '%Y-%m-%d %H:%i:%s')
END,
Degree = NULLIF(@Degree, ''),
DegreeType = NULLIF(@DegreeType, ''),
College = NULLIF(@College, ''),
Division = NULLIF(@Division, ''),
Department = NULLIF(@Department, '');

# BOARD_MEMBERS
LOAD DATA LOCAL INFILE '../board_members.csv'
INTO TABLE board_members
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(@FanID, @Board, @RoleName, @StartDate, @EndDate, @CurrentFlag)
SET
FanID = NULLIF(@FanID, ''),
Board = NULLIF(@Board, ''),
RoleName = NULLIF(@RoleName, ''),
StartDate = CASE
    WHEN NULLIF(@StartDate, '') IS NULL THEN NULL
    ELSE STR_TO_DATE(@StartDate, '%Y-%m-%d')
END,
EndDate = CASE
    WHEN NULLIF(@EndDate, '') IS NULL THEN NULL
    ELSE STR_TO_DATE(@EndDate, '%Y-%m-%d')
END,
CurrentFlag = NULLIF(@CurrentFlag, '');

# WEALTH

LOAD DATA LOCAL INFILE '../wealth.csv'
INTO TABLE wealth
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(@FanID, @IWave_Properties, @IWave_PropertiesValue, @IWave_AnnualCapacity, @IWave_AnnualCapacityTargetProduction)
SET
FanID = NULLIF(@FanID, ''),
IWave_Properties = NULLIF(@IWave_Properties, ''),
IWave_PropertiesValue = NULLIF(@IWave_PropertiesValue, ''),
IWave_AnnualCapacity = NULLIF(@IWave_AnnualCapacity, ''),
IWave_AnnualCapacityTargetProduction = NULLIF(@IWave_AnnualCapacityTargetProduction, '');
