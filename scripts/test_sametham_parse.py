"""Self-check for the Sametham page parser on two cached pages. Run: python scripts/test_sametham_parse.py"""
from pathlib import Path

from fetch_sametham_school_details import RAW_DIR, parse

r = parse((RAW_DIR / "11056.html").read_text(encoding="utf-8"))
assert r["students_std05"] == 34 and r["students_total"] == 207
assert r["students_hss_plus1"] == 91 and r["students_hss_total"] == 184
assert r["staff_school_employees"] == 12 and r["infra_13_total_class_room"] == "10"
r42 = parse((RAW_DIR / "42001.html").read_text(encoding="utf-8"))
assert r42["staff_hss_employees"] == 22 and r42["hss_code"] == "1071" and r["hss_code"] == "14066"
assert r42["vhse_code"] == "NIL" and parse((RAW_DIR / "42006.html").read_text(encoding="utf-8"))["vhse_code"] == "901005"

from build_school_capacity import post_counts
c = post_counts("High School Assistant Hindi 1; High School Asst (lower scale) 1; UP School Assistant Gr I 2; "
                "Teacher (L P School) Gr II 3; P D Teacher 1; Headmaster LP/UP 1; HSST Senior English 2; "
                "Teacher (LP/UP) - Arabic (Grade II) 1; Office Attendant Gr II 1; Drawing Teacher Gr II 1")
assert (c["hs"], c["hs_lower"], c["up"], c["lp"], c["pd"], c["head"], c["hss"], c["lang"], c["non_teaching"],
        c["special"], c["other"]) == (1, 1, 2, 3, 1, 1, 2, 1, 1, 1, 0), c  # HSA Hindi = HS, Arabic (LP/UP) = language
print("ok")
