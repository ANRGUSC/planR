import pytest
import campus_model as cm

def test_total_student_courses_rooms_types():
    assert isinstance(cm.CampusModel().total_students(), int) == True
    assert isinstance(cm.CampusModel().total_courses(), int) == True
    assert isinstance(cm.CampusModel().total_rooms(), int) == True

