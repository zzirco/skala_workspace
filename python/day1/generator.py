import sys

# -------------------------------
# 1) 일반 리스트 사용
# -------------------------------

# 0부터 999,999까지의 숫자를 리스트로 생성
# 리스트는 메모리에 모든 값을 한 번에 저장하기 때문에 메모리 사용량이 큼
numbers_list = [i for i in range(1_000_000)]

# 리스트의 합계 계산 (sum 함수는 내부적으로 반복문을 돌며 합산)
sum_list = sum(numbers_list)
print("리스트 합계:", sum_list)

# -------------------------------
# 2) 제너레이터 함수 사용
# -------------------------------

# 제너레이터 함수 정의
# yield 키워드를 사용하여 값을 하나씩 반환
# → 전체 데이터를 메모리에 올리지 않고, 필요할 때마다 생성
def number_generator(n):
    for i in range(n):
        yield i

# 제너레이터 객체 생성 (리스트와 달리 데이터가 즉시 메모리에 저장되지 않음)
numbers_gen = number_generator(1_000_000)

# 제너레이터의 합계 계산
# sum은 next()를 호출해 제너레이터에서 값을 하나씩 꺼내 합산
sum_gen = sum(numbers_gen)
print("제너레이터 합계:", sum_gen)

# -------------------------------
# 3) 두 방법의 차이 확인
# -------------------------------
print("\n메모리 사용량 비교")
print(f"리스트: {sys.getsizeof(numbers_list)} bytes")
print(f"제너레이터: {sys.getsizeof(numbers_gen)} bytes")
