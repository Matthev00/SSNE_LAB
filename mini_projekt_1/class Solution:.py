class Solution:
    def rotate(self, matrix: list[list[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        row = [0] * n
        new_matrix = [row[:] for i in range(n)]

        for r in range(n):
            for c in range(n):
                new_matrix[c][-(1+r)] = matrix[r][c]


        matrix = new_matrix


if __name__ == "__main__":
    s = Solution()
    matrix = [[1,2,3],[4,5,6],[7,8,9]]
    s.rotate(matrix)
    print(matrix)
    matrix = [[5, 1, 9,11],[2, 4, 8,10],[13, 3, 6, 7],[15,14,12,16]]
    s.rotate(matrix)
    print(matrix)
    matrix = [[1]]
    s.rotate(matrix)
    print(matrix)
    matrix = [[1,2],[3,4]]
    s.rotate(matrix)
    print(matrix)