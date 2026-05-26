select count(*) from base;
select count(*) from board_members;
select count(*) from ten_year_donation;
select count(*) from demographic;
select count(*) from education;
select count(*) from wealth;


#Preparar ten_year_donations
CREATE TABLE ten_year_features AS
SELECT 
    *,
    IF(YearsCount < 5 OR Avg_pct_increase_5years IS NULL, 1, 0) AS insufficient_history
FROM (
    SELECT 
        FanID, DriveYear, Num_group_donating, Last_payment_date, TotalPledgeAnual, L1,
        next_year_donation, MaxYear, MinYear, YearsCount,
        (COALESCE(L1, 0) + COALESCE(L2, 0) + COALESCE(L3, 0)) / 
        NULLIF(
            (CASE WHEN L1 IS NOT NULL THEN 1 ELSE 0 END) +
			(CASE WHEN L2 IS NOT NULL THEN 1 ELSE 0 END) +
			(CASE WHEN L3 IS NOT NULL THEN 1 ELSE 0 END), 
        0) AS Avg_pledges_3years,
        (
            COALESCE(growth_1, 0) + COALESCE(growth_2, 0) + 
            COALESCE(growth_3, 0) + COALESCE(growth_4, 0) + 
            COALESCE(growth_5, 0)
        ) / 
        NULLIF(
            (CASE WHEN growth_1 IS NOT NULL THEN 1 ELSE 0 END) +
            (CASE WHEN growth_2 IS NOT NULL THEN 1 ELSE 0 END) +
            (CASE WHEN growth_3 IS NOT NULL THEN 1 ELSE 0 END) +
            (CASE WHEN growth_4 IS NOT NULL THEN 1 ELSE 0 END) +
            (CASE WHEN growth_5 IS NOT NULL THEN 1 ELSE 0 END), 
        0) AS Avg_pct_increase_5years
    FROM (
        SELECT 
            *,
            ((TotalPledgeAnual - NULLIF(L1,0)) / NULLIF(L1,0) * 100) AS growth_1,
            ((L1 - NULLIF(L2,0)) / NULLIF(L2,0) * 100) AS growth_2,
            ((L2 - NULLIF(L3,0)) / NULLIF(L3,0) * 100) AS growth_3,
            ((L3 - NULLIF(L4,0)) / NULLIF(L4,0) * 100) AS growth_4,
            ((L4 - NULLIF(L5,0)) / NULLIF(L5,0) * 100) AS growth_5
        FROM (
            SELECT 
                FanID, DriveYear, TotalPledgeAnual, Last_payment_date, Num_group_donating,
                MIN(DriveYear) OVER(PARTITION BY FanID) as MinYear,
                MAX(DriveYear) OVER(PARTITION BY FanID) as MaxYear,
                COUNT(DriveYear) OVER(PARTITION BY FanID) as YearsCount,
                LEAD(TotalPledgeAnual) OVER (PARTITION BY FanID ORDER BY DriveYear) AS next_year_donation,
                LAG(TotalPledgeAnual, 1) OVER (PARTITION BY FanID ORDER BY DriveYear) AS L1,
                LAG(TotalPledgeAnual, 2) OVER (PARTITION BY FanID ORDER BY DriveYear) AS L2,
                LAG(TotalPledgeAnual, 3) OVER (PARTITION BY FanID ORDER BY DriveYear) AS L3,
                LAG(TotalPledgeAnual, 4) OVER (PARTITION BY FanID ORDER BY DriveYear) AS L4,
                LAG(TotalPledgeAnual, 5) OVER (PARTITION BY FanID ORDER BY DriveYear) AS L5
            FROM (
                SELECT 
                    FanID, DriveYear,
                    SUM(PledgeAmount) AS TotalPledgeAnual,
                    MAX(MostRecentPaymentPledgeDate) AS Last_payment_date,
                    COUNT(FundGroupName) AS Num_group_donating
                FROM ten_year_donation
                GROUP BY FanID, DriveYear
            ) AS agrupado
        ) AS con_lags
    ) AS con_growths
) AS final_step;

# Preparar education
CREATE TABLE education_final AS  
WITH ranked AS ( 
SELECT
        FanID,
        EducationStatus,
        Degree,
        DegreeType,
        College,
        CASE 
            WHEN Degree LIKE '%Doctor%' THEN 3
            WHEN Degree LIKE '%Master%' THEN 2
            WHEN Degree LIKE '%Bachelor%' THEN 1
            ELSE 0
        END AS DegreeRank
    FROM education
    WHERE DegreeType = 'PRIMARY'
)  
SELECT FanID, EducationStatus, Degree, College
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY FanID ORDER BY DegreeRank DESC) AS rn
    FROM ranked
) t
WHERE rn = 1
ORDER BY FanID asc;

#Preparar Board Members
CREATE TABLE board_members_final AS
SELECT 
    FanID,
    1 AS was_board_member,
    MAX(CurrentFlag) AS CurrentFlag
FROM board_members
GROUP BY FanID;

#Preparar wealth
CREATE TABLE wealth_final AS
SELECT FanID, IWave_Properties, IWave_PropertiesValue, IWave_AnnualCapacityTargetProduction, IF(IWave_PropertiesValue IS NULL, 1, 0) as Properties_null 
FROM wealth;


#Preparar base
CREATE TABLE base_final AS
SELECT 
	FanID,
	state,
	YearsOfDonating,
    HighestMembershipLevel,
    CYAnnualFundPledged,
    CYAnnualFundPaid,
    MostRecentMembershipLevel,
    IsAlumni
FROM base;
                         
#VERIFICATIONS                         
SELECT Avg_pledges_3years, Avg_pct_increase_5years 
FROM ten_year_features 
LIMIT 10;

SELECT COUNT(*) 
FROM ten_year_features t
LEFT JOIN base_final bf ON t.FanID = bf.FanID
WHERE bf.FanID IS NULL;

SELECT DriveYear, COUNT(*) as observaciones
FROM donations_df
GROUP BY DriveYear
ORDER BY DriveYear;

SELECT FanID, DriveYear, TotalPledgeAnual, next_year_donation, COUNT(*) OVER (PARTITION BY DriveYear)
FROM donations_df
WHERE DriveYear = 2025;

#SAVE FINAL DF
DROP TABLE donations_df;

CREATE TABLE donations_df AS 
SELECT 
		t.FanID,
		t.DriveYear,
        t.Num_group_donating,
        COALESCE(t.L1, 0) AS Lag1_Pledge,
		CASE WHEN t.L1 IS NULL THEN 1 ELSE 0 END AS Lag1_null,
        t.TotalPledgeAnual,
        t.next_year_donation,
        t.MaxYear,
        t.MinYear,
        t.YearsCount,
		COALESCE(t.Avg_pledges_3years, 0) AS Avg_pledges_3years,
		COALESCE(t.Avg_pct_increase_5years, 0) AS Avg_pct_increase_5years,
        t.insufficient_history,
        e.EducationStatus,
        e.Degree,
        e.College,
        COALESCE(b.was_board_member, 0) AS was_board_member,
		COALESCE(b.CurrentFlag, 0) AS CurrentFlag,
        COALESCE(w.IWave_Properties, 0) AS IWave_Properties,
		COALESCE(w.IWave_PropertiesValue, 0) AS IWave_PropertiesValue,
        COALESCE(w.IWave_AnnualCapacityTargetProduction, 0) AS IWave_AnnualCapacityTargetProduction,
		COALESCE(w.Properties_null, 1) AS Properties_null,
        d.Age,
        bf.state,
        bf.YearsOfDonating,
        bf.HighestMembershipLevel,
        bf.MostRecentMembershipLevel,
		bf.CYAnnualFundPledged,
		bf.CYAnnualFundPaid,
        bf.IsAlumni
FROM ten_year_features t LEFT JOIN education_final e ON t.FanID = e.FanID 
						 LEFT JOIN board_members_final b ON t.FanID = b.FanID 
                         LEFT JOIN wealth_final w ON t.FanID = w.FanID
                         LEFT JOIN demographic d ON t.FanID = d.FanID
                         LEFT JOIN base_final bf ON t.FanID = bf.FanID
WHERE t.next_year_donation IS NOT NULL AND t.DriveYear BETWEEN 2014 AND 2025;


SELECT Degree, COUNT(*)
FROM donations_df 
GROUP BY Degree 
ORDER BY COUNT(*) DESC;

SELECT state, COUNT(*) as cnt
FROM donations_df
GROUP BY state
ORDER BY cnt DESC;

SELECT * FROM donations_df
INTO OUTFILE 'modeling_table.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';

SELECT * FROM ten_year_features;