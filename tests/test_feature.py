from datetime import datetime

import pytest

from src.feature import Feature



class TestFeature:

    def test_that_take_int_difference_works(self):
        number1, number_2 = 10, 20
        result = Feature.take_int_difference(number1, number_2)
        assert result == 10

    def test_that_datetime_difference_works(self):
        datetime1, datetime2, interval = datetime(2020, 12, 31), datetime(2021, 12, 31), 'D'
        result = Feature.take_datetime_difference_in_years(datetime1, datetime2, interval)
        assert round(result) == 1

    def test_that_wrong_interval_raises_error(self):
        datetime1, datetime2, interval = datetime(2020, 12, 31), datetime(2021, 12, 31), 'M'
        with pytest.raises(ValueError):
            Feature.take_datetime_difference_in_years(datetime1, datetime2, interval)

