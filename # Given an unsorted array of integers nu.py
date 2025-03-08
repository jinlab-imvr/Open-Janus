# Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

# Input: nums = [100,4,200,1,3,2]
# Output: 4
# Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

def find_longest(nums):
    nums = nums.sort()
    
    res = 0
    n = len(nums)
    print(nums)
    if n == 0 or n == 1:
        return n
    tmp_res = 0
    left_pt = 0
    for right_pt in range(1,n):
        if nums[right_pt] == nums[right_pt-1] + 1:
            tmp_res += 1
        else:
            left_pt = right_pt
            tmp_res = 1
            res = max(res,tmp_res)
            
    return res

nums=[100,4,200,1,3,2]
print(nums)
print(find_longest(nums))