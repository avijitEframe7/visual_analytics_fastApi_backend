-- Run on your app database (e.g. employeeinfo).
-- 1) Adds nullable camera_id on exception_logs if missing.
-- 2) Adds FK: exception_logs.camera_id -> dbo.camera(camera_id).
--
-- If the FK step fails, fix orphan rows first (camera_id not in dbo.camera), e.g.:
--   UPDATE dbo.exception_logs SET camera_id = NULL WHERE camera_id IS NOT NULL
--     AND camera_id NOT IN (SELECT camera_id FROM dbo.camera);

USE employeeinfo;
GO

IF COL_LENGTH(N'dbo.exception_logs', N'camera_id') IS NULL
BEGIN
    ALTER TABLE dbo.exception_logs ADD camera_id INT NULL;
END
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.foreign_keys fk
    INNER JOIN sys.tables t ON fk.parent_object_id = t.object_id
    INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE s.name = N'dbo'
      AND t.name = N'exception_logs'
      AND fk.name = N'FK_exception_logs_camera'
)
BEGIN
    ALTER TABLE dbo.exception_logs
    ADD CONSTRAINT FK_exception_logs_camera
    FOREIGN KEY (camera_id) REFERENCES dbo.camera (camera_id);
END
GO
