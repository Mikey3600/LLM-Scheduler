from scheduler.metrics import goodput
def test_goodput_is_highest_rate_meeting_ninety_percent():
    assert goodput([{'rate':10,'slo_attainment':.91},{'rate':20,'slo_attainment':.89}])==10
