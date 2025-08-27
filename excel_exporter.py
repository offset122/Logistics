import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Iterable

import xlsxwriter

# Configure simple logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExcelExporter:
    """Export optimization results to Excel with multiple sheets (robust/fail-safe)."""

    def __init__(self):
        self.workbook: xlsxwriter.Workbook | None = None
        self.worksheet_formats = {}

    # --- Public API ------------------------------------------------------
    def export_all_results(
        self,
        run_info: Dict[str, Any],
        routes: List[Dict[str, Any]],
        metro_schedules: List[Dict[str, Any]],
        filepath: str,
    ):
        """Export all results to Excel file with multiple sheets (safe)."""
        try:
            # Normalize inputs to lists (avoid None)
            routes = list(routes or [])
            metro_schedules = list(metro_schedules or [])

            # Create workbook
            self.workbook = xlsxwriter.Workbook(filepath)
            self._setup_formats()

            # Sheet 1: Summary
            self._create_summary_sheet(run_info or {}, routes, metro_schedules)

            # Sheet 2: Drone Routes
            self._create_drone_routes_sheet(routes)

            # Sheet 3: Truck Routes
            self._create_truck_routes_sheet(routes)

            # Sheet 4: Metro Timetables (Up)
            self._create_metro_timetable_sheet(metro_schedules, "up")

            # Sheet 5: Metro Timetables (Down)
            self._create_metro_timetable_sheet(metro_schedules, "down")

            # Sheet 6: Shipments per Train
            self._create_shipments_sheet(metro_schedules)

            # Sheet 7: Runtime Comparison
            self._create_runtime_comparison_sheet(run_info or {})

        except Exception as e:
            logger.exception("Failed while exporting Excel results: %s", e)
            raise
        finally:
            if self.workbook:
                self.workbook.close()

    # --- Helpers --------------------------------------------------------
    def _ensure_dict(self, value: Any) -> Dict[str, Any]:
        """Return a dict if possible. If value is a JSON string, parse it. Otherwise return {}."""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            # Try parsing JSON; if fails, return empty dict and log
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
                logger.debug("Parsed JSON is not a dict: %s", parsed)
            except json.JSONDecodeError:
                logger.debug("Details string is not JSON: %s", value[:200])
            return {}
        # For other types, log and return {}
        logger.debug("Unexpected details type: %s (%s)", type(value), value)
        return {}

    def _setup_formats(self):
        """Setup cell formats for the workbook."""
        assert self.workbook is not None, "Workbook not created"
        self.worksheet_formats = {
            "header": self.workbook.add_format(
                {"bold": True, "font_size": 12, "bg_color": "#D7E4BC", "border": 1}
            ),
            "subheader": self.workbook.add_format(
                {"bold": True, "font_size": 10, "bg_color": "#F2F2F2", "border": 1}
            ),
            "data": self.workbook.add_format({"font_size": 10, "border": 1}),
            "number": self.workbook.add_format(
                {"font_size": 10, "border": 1, "num_format": "#,##0.00"}
            ),
            "time": self.workbook.add_format(
                {"font_size": 10, "border": 1, "num_format": "[h]:mm"}
            ),
        }

    # --- Sheet builders -------------------------------------------------
    def _create_summary_sheet(
        self, run_info: Dict[str, Any], routes: List[Dict[str, Any]], metro_schedules: List[Dict[str, Any]]
    ):
        worksheet = self.workbook.add_worksheet("Summary")
        row = 0

        # Title (use merge safely)
        worksheet.merge_range(
            row, 0, row, 3, "Branch-and-Price Optimization Results", self.worksheet_formats["header"]
        )
        row += 2

        # Run Information
        worksheet.write(row, 0, "Run Information", self.worksheet_formats["subheader"])
        row += 1

        info_data = [
            ["Run ID", run_info.get("id", "N/A")],
            ["Timestamp", run_info.get("timestamp", "N/A")],
            ["Status", run_info.get("status", "N/A")],
            ["Objective Value", run_info.get("objective_value", 0.0)],
            ["Runtime (seconds)", run_info.get("runtime", 0.0)],
            ["Iterations", run_info.get("iterations", 0)],
        ]

        for info_row in info_data:
            worksheet.write(row, 0, info_row[0], self.worksheet_formats["data"])
            worksheet.write(row, 1, info_row[1], self.worksheet_formats["data"])
            row += 1

        row += 1

        # Solution Summary
        worksheet.write(row, 0, "Solution Summary", self.worksheet_formats["subheader"])
        row += 1

        # Count routes by type safely
        drone_routes = [r for r in routes if (r.get("vehicle_type") or "").lower() == "drone"]
        truck_routes = [r for r in routes if (r.get("vehicle_type") or "").lower() == "truck"]
        up_schedules = [s for s in metro_schedules if (s.get("direction") or "up").lower() == "up"]
        down_schedules = [s for s in metro_schedules if (s.get("direction") or "down").lower() == "down"]

        # total cost safe-sum
        total_cost_routes = sum(self._safe_number(r.get("cost", 0.0)) for r in routes)
        total_cost_metro = sum(self._safe_number(s.get("cost", 0.0)) for s in metro_schedules)

        summary_data = [
            ["Total Drone Routes", len(drone_routes)],
            ["Total Truck Routes", len(truck_routes)],
            ["Metro Up-line Schedules", len(up_schedules)],
            ["Metro Down-line Schedules", len(down_schedules)],
            ["Total Cost", total_cost_routes + total_cost_metro],
        ]

        for summary_row in summary_data:
            worksheet.write(row, 0, summary_row[0], self.worksheet_formats["data"])
            worksheet.write(row, 1, summary_row[1], self.worksheet_formats["data"])
            row += 1

        # Auto-fit columns
        worksheet.set_column(0, 0, 30)
        worksheet.set_column(1, 1, 20)

    def _create_drone_routes_sheet(self, routes: List[Dict[str, Any]]):
        worksheet = self.workbook.add_worksheet("Drone Routes")
        drone_routes = [r for r in routes if (r.get("vehicle_type") or "").lower() == "drone"]

        if not drone_routes:
            worksheet.write(0, 0, "No drone routes in solution", self.worksheet_formats["data"])
            return

        headers = [
            "Route ID",
            "Path",
            "Total Distance (km)",
            "Flight Time (min)",
            "Energy Consumption",
            "Load",
            "Cost",
            "Selection Value",
        ]
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, self.worksheet_formats["header"])

        for row_idx, route in enumerate(drone_routes, start=1):
            # ensure details is dict
            details_raw = route.get("details", {})  # might be dict or JSON string
            details = self._ensure_dict(details_raw)

            path_list = details.get("route", details.get("path", []))
            if isinstance(path_list, str):
                # path_list might be a CSV string like "1,2,3" - convert to list
                path_list = [p.strip() for p in path_list.split(",") if p.strip()]

            data = [
                route.get("route_id", route.get("id", f"Route_{row_idx}")),
                " -> ".join(map(str, path_list)),
                self._safe_number(details.get("total_distance", 0.0)),
                self._safe_number(details.get("flight_time", 0.0)),
                self._safe_number(details.get("energy_consumption", 0.0)),
                self._safe_number(details.get("load", 0)),
                self._safe_number(route.get("cost", 0.0)),
                self._safe_number(route.get("selection_value", 0.0)),
            ]

            for col, value in enumerate(data):
                # numeric columns indices: 2,3,4,6,7
                if col in (2, 3, 4, 6, 7):
                    worksheet.write(row_idx, col, value, self.worksheet_formats["number"])
                else:
                    worksheet.write(row_idx, col, value, self.worksheet_formats["data"])

        for col in range(len(headers)):
            worksheet.set_column(col, col, 18)

    def _create_truck_routes_sheet(self, routes: List[Dict[str, Any]]):
        worksheet = self.workbook.add_worksheet("Truck Routes")
        truck_routes = [r for r in routes if (r.get("vehicle_type") or "").lower() == "truck"]

        if not truck_routes:
            worksheet.write(0, 0, "No truck routes in solution", self.worksheet_formats["data"])
            return

        headers = ["Route ID", "Path", "Total Distance (km)", "Total Time (min)", "Load", "Cost", "Selection Value"]
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, self.worksheet_formats["header"])

        for row_idx, route in enumerate(truck_routes, start=1):
            details_raw = route.get("details", {})
            details = self._ensure_dict(details_raw)

            path_list = details.get("route", details.get("path", []))
            if isinstance(path_list, str):
                path_list = [p.strip() for p in path_list.split(",") if p.strip()]

            data = [
                route.get("route_id", route.get("id", f"Route_{row_idx}")),
                " -> ".join(map(str, path_list)),
                self._safe_number(details.get("total_distance", 0.0)),
                self._safe_number(details.get("total_time", 0.0)),
                self._safe_number(details.get("load", 0)),
                self._safe_number(route.get("cost", 0.0)),
                self._safe_number(route.get("selection_value", 0.0)),
            ]

            for col, value in enumerate(data):
                if col in (2, 3, 5, 6):
                    worksheet.write(row_idx, col, value, self.worksheet_formats["number"])
                else:
                    worksheet.write(row_idx, col, value, self.worksheet_formats["data"])

        for col in range(len(headers)):
            worksheet.set_column(col, col, 18)

    def _create_metro_timetable_sheet(self, metro_schedules: List[Dict[str, Any]], direction: str):
        sheet_name = f"Metro {direction.title()}-line"
        worksheet = self.workbook.add_worksheet(sheet_name)

        direction_schedules = [s for s in metro_schedules if (s.get("direction") or "up").lower() == direction.lower()]

        if not direction_schedules:
            worksheet.write(0, 0, f"No {direction}-line schedules in solution", self.worksheet_formats["data"])
            return

        headers = ["Schedule ID", "Departure Time", "Arrival Time", "Load", "Cost", "Selection Value"]
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, self.worksheet_formats["header"])

        for row_idx, schedule in enumerate(direction_schedules, start=1):
            # schedule may be dict or JSON string
            schedule = self._ensure_schedule(schedule)

            data = [
                schedule.get("id", f"Schedule_{row_idx}"),
                self._minutes_to_time_str(schedule.get("departure", 0)),
                self._minutes_to_time_str(schedule.get("arrival", 0)),
                self._safe_number(schedule.get("load", 0)),
                self._safe_number(schedule.get("cost", 0.0)),
                self._safe_number(schedule.get("selection_value", 0.0)),
            ]

            for col, value in enumerate(data):
                if col in (4, 5):
                    worksheet.write(row_idx, col, value, self.worksheet_formats["number"])
                else:
                    worksheet.write(row_idx, col, value, self.worksheet_formats["data"])

        for col in range(len(headers)):
            worksheet.set_column(col, col, 18)

    def _create_shipments_sheet(self, metro_schedules: List[Dict[str, Any]]):
        worksheet = self.workbook.add_worksheet("Shipments per Train")

        if not metro_schedules:
            worksheet.write(0, 0, "No metro schedules in solution", self.worksheet_formats["data"])
            return

        headers = ["Train ID", "Direction", "Departure Time", "Total Shipments", "Average Load", "Total Cost"]
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, self.worksheet_formats["header"])

        # Group schedules by train/direction (robust)
        train_groups: Dict[str, List[Dict[str, Any]]] = {}
        for schedule_raw in metro_schedules:
            schedule = self._ensure_schedule(schedule_raw)
            direction = (schedule.get("direction") or "up").lower()
            departure = int(schedule.get("departure", 0))
            train_key = f"{direction}_{departure // 60}"  # group by hour safely
            train_groups.setdefault(train_key, []).append(schedule)

        row = 1
        for train_key, schedules in train_groups.items():
            direction = train_key.split("_")[0]
            total_shipments = sum(self._safe_number(s.get("load", 0)) for s in schedules)
            avg_load = total_shipments / len(schedules) if schedules else 0
            total_cost = sum(self._safe_number(s.get("cost", 0)) for s in schedules)
            departure_time = min(int(s.get("departure", 0)) for s in schedules) if schedules else 0

            data = [
                train_key,
                direction.title(),
                self._minutes_to_time_str(departure_time),
                total_shipments,
                avg_load,
                total_cost,
            ]

            for col, value in enumerate(data):
                if col in (3, 4, 5):
                    worksheet.write(row, col, value, self.worksheet_formats["number"])
                else:
                    worksheet.write(row, col, value, self.worksheet_formats["data"])
            row += 1

        for col in range(len(headers)):
            worksheet.set_column(col, col, 18)

    def _create_runtime_comparison_sheet(self, run_info: Dict[str, Any]):
        worksheet = self.workbook.add_worksheet("Runtime Comparison")
        headers = ["Algorithm", "Type", "Runtime (seconds)", "Columns Generated", "Quality"]
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, self.worksheet_formats["header"])

        # Attempt to populate from run_info if available (safe access)
        algs = run_info.get("algorithm_stats") or []
        if algs and isinstance(algs, list):
            for r_idx, alg in enumerate(algs, start=1):
                worksheet.write(r_idx, 0, alg.get("name", "Unknown"), self.worksheet_formats["data"])
                worksheet.write(r_idx, 1, alg.get("type", "N/A"), self.worksheet_formats["data"])
                worksheet.write(r_idx, 2, self._safe_number(alg.get("runtime", 0.0)), self.worksheet_formats["number"])
                worksheet.write(r_idx, 3, int(alg.get("columns_generated", 0)), self.worksheet_formats["number"])
                worksheet.write(r_idx, 4, str(alg.get("quality", "N/A")), self.worksheet_formats["data"])
        else:
            # default example rows
            algorithms = [
                ["ULA-T", "Basic Truck", 0.0, 0, "N/A"],
                ["ULA-D", "Basic Drone", 0.0, 0, "N/A"],
                ["SELA-M", "Basic Metro", 0.0, 0, "N/A"],
                ["BLA-T", "Optimized Truck", 0.0, 0, "N/A"],
                ["BLA-D", "Optimized Drone", 0.0, 0, "N/A"],
                ["BALA-M", "Optimized Metro", 0.0, 0, "N/A"],
            ]
            for row_idx, alg_data in enumerate(algorithms, start=1):
                for col, value in enumerate(alg_data):
                    if col == 2:
                        worksheet.write(row_idx, col, value, self.worksheet_formats["number"])
                    else:
                        worksheet.write(row_idx, col, value, self.worksheet_formats["data"])

        for col in range(len(headers)):
            worksheet.set_column(col, col, 20)

    # --- Utility methods -----------------------------------------------
    def _minutes_to_time_str(self, minutes: int) -> str:
        """Convert minutes since midnight to HH:MM format (safe)."""
        try:
            minutes = int(minutes)
        except Exception:
            return "00:00"
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"

    def _safe_number(self, value: Any) -> float:
        """Convert value to float safely (fallback to 0.0)."""
        try:
            return float(value)
        except Exception:
            return 0.0

    def _ensure_schedule(self, schedule_raw: Any) -> Dict[str, Any]:
        """Ensure a schedule record is a dict (parse JSON string if necessary)."""
        if isinstance(schedule_raw, dict):
            return schedule_raw
        if isinstance(schedule_raw, str):
            try:
                parsed = json.loads(schedule_raw)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                logger.debug("Schedule string not JSON: %s", schedule_raw[:200])
        logger.debug("Unexpected schedule type: %s", type(schedule_raw))
        return {}

