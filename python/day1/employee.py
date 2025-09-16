# 직원들의 정보를 담은 리스트 (각 직원은 dict 형태)
employees = [
    {"name": "Alice", "department": "Engineering", "age": 30, "salary": 85000},
    {"name": "Bob", "department": "Marketing", "age": 25, "salary": 60000},
    {"name": "Charlie", "department": "Engineering", "age": 35, "salary": 95000},
    {"name": "David", "department": "HR", "age": 45, "salary": 70000},
    {"name": "Eve", "department": "Engineering", "age": 28, "salary": 78000},
]

# ------------------------------------------------------------------------
# 1) 부서가 "Engineering"이고 salary >= 80000인 직원들의 이름만 리스트로 출력
# - employees 리스트에서 조건을 만족하는 직원만 필터링
# - 리스트 컴프리헨션을 사용하여 emp["name"]만 추출
engineering_high_salary = [
    emp["name"] for emp in employees
    if emp["department"] == "Engineering" and emp["salary"] >= 80000
]
print("조건1:", engineering_high_salary)

# ------------------------------------------------------------------------
# 2) 30세 이상인 직원의 이름과 부서를 튜플 (name, department) 형태로 리스트로 출력
# - employees 리스트를 순회하면서 age >= 30 조건을 만족하는 직원만 필터링
filtered_over_30 = [emp for emp in employees if emp["age"] >= 30]

# - zip을 사용하여 이름 리스트와 부서 리스트를 묶어 튜플 형태로 변환
over_30 = list(zip(
    [emp["name"] for emp in filtered_over_30],       # 이름만 추출한 리스트
    [emp["department"] for emp in filtered_over_30]  # 부서만 추출한 리스트
))
print("조건2:", over_30)

# ------------------------------------------------------------------------
# 3) 급여(salary) 기준으로 직원 리스트를 내림차순 정렬하고, 상위 3명의 이름과 급여 출력
# - sorted() 함수 사용: key=lambda x: x["salary"] → 직원 dict에서 salary 값을 기준으로 정렬
# - reverse=True → 내림차순 정렬
# - [:3] → 정렬된 리스트 중 상위 3명만 선택
top_3_salary = sorted(employees, key=lambda x: x["salary"], reverse=True)[:3]

# - zip을 사용하여 이름 리스트와 급여 리스트를 묶어 (이름, 급여) 튜플 리스트 생성
top_3_salary_info = list(zip(
    [emp["name"] for emp in top_3_salary],    # 상위 3명의 이름 리스트
    [emp["salary"] for emp in top_3_salary]   # 상위 3명의 급여 리스트
))
print("조건3:", top_3_salary_info)
