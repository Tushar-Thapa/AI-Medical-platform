from modules.lab_report.service import LabReportService
from shared.schemas.lab_report import LabReportRequest

service = LabReportService()

req = LabReportRequest(
    patient_id="p1",
    report_base64="dummy_base64"
)

response = service.run(req)
print(response)