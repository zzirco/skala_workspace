import sys
import time

# -------------------------------
# 제너레이터 함수 정의
# -------------------------------
# 0 이상 n 미만의 정수 중에서 짝수만 제곱하여 하나씩 생성
def even_square_gen(n):
    for i in range(0, n, 2):  # range(0, n, 2) → 0부터 n-1까지 2씩 증가
        yield i * i           # 짝수를 제곱한 값을 yield로 하나씩 반환


# -------------------------------
# 리스트 방식 합계와 메모리 사용량 측정 함수
# -------------------------------
def list_sum(n):
    start_time = time.time()  # 시작 시각 기록

    # 리스트 컴프리헨션으로 짝수 제곱 리스트 생성
    numbers = [i * i for i in range(0, n, 2)]

    total = sum(numbers)      # 합계 계산
    end_time = time.time()    # 종료 시각 기록

    elapsed = end_time - start_time  # 처리 시간
    memory = sys.getsizeof(numbers)  # 리스트 메모리 사용량

    return total, elapsed, memory


# -------------------------------
# 제너레이터 방식 합계와 메모리 사용량 측정 함수
# -------------------------------
def generator_sum(n):
    start_time = time.time()  # 시작 시각 기록

    gen = even_square_gen(n)  # 제너레이터 객체 생성
    total = sum(gen)          # 제너레이터의 값을 하나씩 꺼내 합산
    end_time = time.time()    # 종료 시각 기록

    elapsed = end_time - start_time  # 처리 시간
    memory = sys.getsizeof(gen)      # 제너레이터 객체 자체의 크기

    return total, elapsed, memory


# -------------------------------
# 리스트 vs 제너레이터 비교 함수
# -------------------------------
def compare_methods(n):
    # 1) 리스트 방식
    sum_list, time_list, mem_list = list_sum(n)
    print(f"[리스트 방식] 합계={sum_list}, 시간={time_list:.4f}초, 메모리={mem_list} bytes")

    # 2) 제너레이터 방식
    sum_gen, time_gen, mem_gen = generator_sum(n)
    print(f"[제너레이터 방식] 합계={sum_gen}, 시간={time_gen:.4f}초, 메모리={mem_gen} bytes")

    # 3) 메모리와 속도 비교
    print("\n[메모리 절약률] 리스트 대비 제너레이터 메모리 {0:.2f}% 사용".format(mem_gen / mem_list * 100))
    print("[시간 차이] 리스트 {0:.4f}초 vs 제너레이터 {1:.4f}초".format(time_list, time_gen))


# -------------------------------
# main 함수
# -------------------------------
if __name__ == "__main__":
    compare_methods(1_000_001)  # 0부터 1,000,000까지 짝수 제곱 합
