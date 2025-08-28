import pytest
import pandas as pd
from src.dataset import Dataset

@pytest.fixture
def sample_claims_data():
    """Sample claims data for testing"""
    return pd.DataFrame({
        'customer_id': [1, 1, 2, 2, 2, 3, 4, 4, 5],
        'product_type': ['A', 'A', 'B', 'B', 'A', 'C', 'A', 'B', 'C'],
        'claim_amount': [100, 150, 200, 75, 300, 50, 125, 175, 225],
        'claim_date': ['2023-01-01', '2023-01-15', '2023-02-01',
                       '2023-02-15', '2023-03-01', '2023-03-15',
                       '2023-04-01', '2023-04-15', '2023-05-01']
    })


@pytest.fixture
def sample_data():
    """Sample main data for testing"""
    return pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5, 6],
        'product_type': ['A', 'B', 'C', 'A', 'C', 'B'],
        'age': [25, 35, 45, 55, 65, 75],
        'premium': [1000, 1500, 2000, 1200, 1800, 1100],
        'region': ['North', 'South', 'East', 'West', 'North', 'South']
    })


@pytest.fixture
def claims_csv_file(tmp_path, sample_claims_data):
    """Create a temporary CSV file with claims data"""
    file_path = tmp_path / "claims.csv"
    sample_claims_data.to_csv(file_path, sep=';', index=False)
    return str(file_path)


@pytest.fixture
def data_csv_file(tmp_path, sample_data):
    """Create a temporary CSV file with main data"""
    file_path = tmp_path / "data.csv"
    sample_data.to_csv(file_path, sep=';', index=False)
    return str(file_path)


@pytest.fixture
def dataset_instance(data_csv_file, claims_csv_file):
    """Create a Dataset instance for testing"""
    return Dataset(data_csv_file, claims_csv_file)


@pytest.fixture
def empty_claims_data():
    """Empty claims data for edge case testing"""
    return pd.DataFrame(columns=['customer_id', 'product_type', 'claim_amount'])


@pytest.fixture
def empty_data():
    """Empty main data for edge case testing"""
    return pd.DataFrame(columns=['customer_id', 'product_type', 'age', 'premium'])


class TestDatasetInit:
    """Test Dataset initialization"""

    def test_init_success(self, dataset_instance, sample_claims_data, sample_data):
        """Test successful initialization"""
        pd.testing.assert_frame_equal(dataset_instance.claims, sample_claims_data)
        pd.testing.assert_frame_equal(dataset_instance.data, sample_data)
        assert dataset_instance.grouped_claims is None
        assert dataset_instance.dataset is None
        assert dataset_instance.train is None
        assert dataset_instance.test is None

    def test_init_file_not_found(self, tmp_path):
        """Test initialization with non-existent files"""
        with pytest.raises(FileNotFoundError):
            Dataset("non_existent_data.csv", "non_existent_claims.csv")

    def test_init_with_invalid_delimiter(self, tmp_path):
        """Test initialization with wrong delimiter in CSV files"""

        invalid_file = tmp_path / "invalid.csv"
        pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}).to_csv(invalid_file, index=False)

        dataset = Dataset(str(invalid_file), str(invalid_file))
        assert len(dataset.data.columns) == 1


class TestGroupClaims:
    """Test group_claims method"""

    def test_group_claims_single_column(self, dataset_instance):
        """Test grouping by single column"""
        result = dataset_instance.group_claims(['customer_id'], 'claim_amount')

        assert result is dataset_instance  # Check method chaining
        assert dataset_instance.grouped_claims is not None

        expected = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'claims_frequency': [2, 3, 1, 2, 1]
        })

        pd.testing.assert_frame_equal(
            dataset_instance.grouped_claims.sort_values('customer_id').reset_index(drop=True),
            expected
        )

    def test_group_claims_multiple_columns(self, dataset_instance):
        """Test grouping by multiple columns"""
        result = dataset_instance.group_claims(['customer_id', 'product_type'], 'claim_amount')

        assert result is dataset_instance
        assert len(dataset_instance.grouped_claims) == 7  # Unique combinations
        assert 'claims_frequency' in dataset_instance.grouped_claims.columns
        assert set(dataset_instance.grouped_claims.columns) == {'customer_id', 'product_type', 'claims_frequency'}

    def test_group_claims_nonexistent_column(self, dataset_instance):
        """Test grouping with non-existent column"""
        with pytest.raises(KeyError):
            dataset_instance.group_claims(['nonexistent_column'], 'claim_amount')

    def test_group_claims_empty_grouping_columns(self, dataset_instance):
        """Test grouping with empty grouping columns list"""
        with pytest.raises((ValueError, KeyError)):
            dataset_instance.group_claims([], 'claim_amount')

    def test_group_claims_nonexistent_aggregation_column(self, dataset_instance):
        """Test grouping with non-existent aggregation column"""
        with pytest.raises(KeyError):
            dataset_instance.group_claims(['customer_id'], 'nonexistent_column')

    def test_group_claims_with_nulls(self, tmp_path):
        """Test grouping with null values"""
        claims_with_nulls = pd.DataFrame({
            'customer_id': [1, 1, None, 2, 2],
            'product_type': ['A', None, 'B', 'B', 'B'],
            'claim_amount': [100, 150, 200, 75, 300]
        })

        claims_file = tmp_path / "claims_nulls.csv"
        claims_with_nulls.to_csv(claims_file, sep=';', index=False)

        data_file = tmp_path / "data_nulls.csv"
        pd.DataFrame({'customer_id': [1, 2], 'age': [25, 35]}).to_csv(data_file, sep=';', index=False)

        dataset = Dataset(str(data_file), str(claims_file))
        dataset.group_claims(['customer_id'], 'claim_amount')


        assert dataset.grouped_claims is not None
        assert len(dataset.grouped_claims) == 2  # Only non-null customer_ids


class TestCreateDataset:
    """Test create_dataset method"""

    def test_create_dataset_success(self, dataset_instance):
        """Test successful dataset creation after grouping"""
        dataset_instance.group_claims(['customer_id'], 'claim_amount')
        result = dataset_instance.create_dataset(['customer_id'])

        assert result is dataset_instance  # Check method chaining
        assert dataset_instance.dataset is not None
        assert len(dataset_instance.dataset) == 6  # Same as original data
        assert 'claims_frequency' in dataset_instance.dataset.columns

        customer_1_row = dataset_instance.dataset[dataset_instance.dataset['customer_id'] == 1]
        assert customer_1_row['claims_frequency'].iloc[0] == 2

    def test_create_dataset_without_grouping(self, dataset_instance):
        """Test dataset creation without prior grouping"""
        with pytest.raises(TypeError, match="Can only merge Series or DataFrame objects"):
            dataset_instance.create_dataset(['customer_id'])

    def test_create_dataset_nonexistent_merge_column(self, dataset_instance):
        """Test dataset creation with non-existent merge column"""
        dataset_instance.group_claims(['customer_id'], 'claim_amount')

        with pytest.raises(KeyError):
            dataset_instance.create_dataset(['nonexistent_column'])

    def test_create_dataset_multiple_merge_columns(self, dataset_instance):
        """Test dataset creation with multiple merge columns"""
        dataset_instance.group_claims(['customer_id', 'product_type'], 'claim_amount')
        result = dataset_instance.create_dataset(['customer_id', 'product_type'])

        assert result is dataset_instance
        assert dataset_instance.dataset is not None
        assert 'claims_frequency' in dataset_instance.dataset.columns

    def test_create_dataset_left_join_behavior(self, dataset_instance):
        """Test that create_dataset performs left join correctly"""
        dataset_instance.group_claims(['customer_id'], 'claim_amount')
        dataset_instance.create_dataset(['customer_id'])

        # Customer 6 exists in data but not in claims, should have NaN for claims_frequency
        customer_6_row = dataset_instance.dataset[dataset_instance.dataset['customer_id'] == 6]
        assert len(customer_6_row) == 1
        assert pd.isna(customer_6_row['claims_frequency'].iloc[0])

    def test_create_dataset_empty_grouped_claims(self, tmp_path, sample_data):
        """Test dataset creation with empty grouped claims"""
        empty_claims = pd.DataFrame(columns=['customer_id', 'claim_amount'])
        claims_file = tmp_path / "empty_claims.csv"
        empty_claims.to_csv(claims_file, sep=';', index=False)

        data_file = tmp_path / "data.csv"
        sample_data.to_csv(data_file, sep=';', index=False)

        dataset = Dataset(str(data_file), str(claims_file))
        dataset.group_claims(['customer_id'], 'claim_amount')
        dataset.create_dataset(['customer_id'])

        assert dataset.dataset['claims_frequency'].isna().all()


class TestSplitDataset:
    """Test split_dataset method"""

    def test_split_dataset_success(self, dataset_instance):
        """Test successful dataset splitting"""

        dataset_instance.group_claims(['customer_id'], 'claim_amount')
        dataset_instance.create_dataset(['customer_id'])

        train, test = dataset_instance.split_dataset(0.3)

        assert train is not None
        assert test is not None
        assert len(train) + len(test) == len(dataset_instance.dataset)

        total_size = len(dataset_instance.dataset)
        test_ratio = len(test) / total_size
        assert 0.2 <= test_ratio <= 0.4  # Allow some flexibility around 0.3


        pd.testing.assert_frame_equal(dataset_instance.train, train)
        pd.testing.assert_frame_equal(dataset_instance.test, test)

    def test_split_dataset_without_dataset(self, dataset_instance):
        """Test splitting without creating dataset first"""
        with pytest.raises(TypeError, match="Expected sequence or array-like"):
            dataset_instance.split_dataset(0.3)

    def test_split_dataset_invalid_ratio(self, dataset_instance):
        """Test splitting with invalid test ratio"""
        dataset_instance.group_claims(['customer_id'], 'claim_amount')
        dataset_instance.create_dataset(['customer_id'])

        with pytest.raises(ValueError):
            dataset_instance.split_dataset(-0.1)

        with pytest.raises(ValueError):
            dataset_instance.split_dataset(1.1)

    def test_split_dataset_edge_ratios(self, dataset_instance):
        """Test splitting with edge case ratios"""
        dataset_instance.group_claims(['customer_id'], 'claim_amount')
        dataset_instance.create_dataset(['customer_id'])

        train, test = dataset_instance.split_dataset(0.17)  # 1/6 ≈ 0.17, should give 1 test sample
        assert len(train) + len(test) == len(dataset_instance.dataset)
        assert len(test) >= 1  # At least 1 sample in test
        assert len(train) >= 1  # At least 1 sample in train

        train, test = dataset_instance.split_dataset(0.83)  # Should give 5 test, 1 train
        assert len(train) + len(test) == len(dataset_instance.dataset)
        assert len(train) >= 1  # At least 1 sample in train
        assert len(test) >= 1  # At least 1 sample in test

    def test_split_dataset_sklearn_constraints(self, dataset_instance):
        """Test sklearn's specific constraints on train_test_split"""
        dataset_instance.group_claims(['customer_id'], 'claim_amount')
        dataset_instance.create_dataset(['customer_id'])

        with pytest.raises(Exception):
            dataset_instance.split_dataset(0.0)

        with pytest.raises(Exception):
            dataset_instance.split_dataset(1.0)

    def test_split_dataset_no_shuffle(self, dataset_instance):
        """Test that split maintains order (shuffle=False)"""
        dataset_instance.group_claims(['customer_id'], 'claim_amount')
        dataset_instance.create_dataset(['customer_id'])

        original_order = dataset_instance.dataset['customer_id'].tolist()
        train, test = dataset_instance.split_dataset(0.33)

        combined_order = train['customer_id'].tolist() + test['customer_id'].tolist()
        assert combined_order == original_order

    def test_split_dataset_single_row(self, tmp_path):
        """Test splitting with single row dataset - expect appropriate error"""
        single_row_data = pd.DataFrame({
            'customer_id': [1],
            'age': [25]
        })
        single_row_claims = pd.DataFrame({
            'customer_id': [1],
            'claim_amount': [100]
        })

        data_file = tmp_path / "single_data.csv"
        claims_file = tmp_path / "single_claims.csv"
        single_row_data.to_csv(data_file, sep=';', index=False)
        single_row_claims.to_csv(claims_file, sep=';', index=False)

        dataset = Dataset(str(data_file), str(claims_file))
        dataset.group_claims(['customer_id'], 'claim_amount')
        dataset.create_dataset(['customer_id'])

        with pytest.raises(ValueError, match="the resulting train set will be empty"):
            dataset.split_dataset(0.5)

    def test_split_dataset_two_rows(self, tmp_path):
        """Test splitting with two rows dataset - minimum for valid split"""
        two_row_data = pd.DataFrame({
            'customer_id': [1, 2],
            'age': [25, 35]
        })
        two_row_claims = pd.DataFrame({
            'customer_id': [1, 2],
            'claim_amount': [100, 200]
        })

        data_file = tmp_path / "two_data.csv"
        claims_file = tmp_path / "two_claims.csv"
        two_row_data.to_csv(data_file, sep=';', index=False)
        two_row_claims.to_csv(claims_file, sep=';', index=False)

        dataset = Dataset(str(data_file), str(claims_file))
        dataset.group_claims(['customer_id'], 'claim_amount')
        dataset.create_dataset(['customer_id'])

        train, test = dataset.split_dataset(0.5)
        assert len(train) == 1
        assert len(test) == 1
        assert len(train) + len(test) == 2


class TestDatasetIntegration:
    """Integration tests for the complete workflow"""

    def test_full_workflow(self, dataset_instance):
        """Test complete workflow: group -> create -> split"""

        dataset_instance.group_claims(['customer_id'], 'claim_amount')
        dataset_instance.create_dataset(['customer_id'])
        train, test = dataset_instance.split_dataset(0.4)


        assert dataset_instance.grouped_claims is not None
        assert dataset_instance.dataset is not None
        assert dataset_instance.train is not None
        assert dataset_instance.test is not None


        assert len(train) + len(test) == len(dataset_instance.dataset)
        assert 'claims_frequency' in train.columns
        assert 'claims_frequency' in test.columns

    def test_method_chaining(self, dataset_instance):
        """Test that methods can be chained together"""
        result = (dataset_instance
                  .group_claims(['customer_id'], 'claim_amount')
                  .create_dataset(['customer_id']))

        assert result is dataset_instance
        assert dataset_instance.grouped_claims is not None
        assert dataset_instance.dataset is not None

    def test_workflow_with_complex_grouping(self, dataset_instance):
        """Test workflow with multiple grouping columns"""
        dataset_instance.group_claims(['customer_id', 'product_type'], 'claim_amount')
        dataset_instance.create_dataset(['customer_id', 'product_type'])
        train, test = dataset_instance.split_dataset(0.33)

        assert len(dataset_instance.grouped_claims) > 0
        assert len(dataset_instance.dataset) == 6
        assert len(train) + len(test) == 6


