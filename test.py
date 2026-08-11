class Matrix2D:
    def __init__(self, data):
        self.data = data  # assume `data` is a list of lists

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key

            # Handle row slicing/indexing
            if isinstance(row_key, slice):
                rows = self.data[row_key]
            else:
                rows = [self.data[row_key]]

            # Now handle col slicing/indexing for each selected row
            result = []
            for row in rows:
                if isinstance(col_key, slice):
                    result.append(row[col_key])
                else:
                    result.append(row[col_key])

            # If both row and col are int, return single value
            if not isinstance(row_key, slice) and not isinstance(col_key, slice):
                return result[0]
            return result

        else:
            # If only a single index is given, return that row
            return self.data[key]

# Example usage
matrix = Matrix2D([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix[:1, 2])
print(matrix[1:, 2])      # 6  -> single element
print(matrix[0:2, 1])    # [2, 5]  -> slice rows, single column
print(matrix[1, 0:2])    # [4, 5]  -> single row, slice columns
print(matrix[0:2, 0:2])  # [[1, 2], [4, 5]]
print(matrix[1])         # [4, 5, 6]  -> just a row
