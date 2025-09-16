from collections import defaultdict

employees = [
    {"name": "Alice", "department": "Engineering", "age": 30, "salary": 85000},
    {"name": "Bob", "department": "Marketing", "age": 25, "salary": 60000},
    {"name": "Charlie", "department": "Engineering", "age": 35, "salary": 95000},
    {"name": "David", "department": "HR", "age": 45, "salary": 70000},
    {"name": "Eve", "department": "Engineering", "age": 28, "salary": 78000},
]

# ------------------------------------------------------------------------
# 추가문제 : 모든 부서별 평균 급여 출력
# - defaultdict(list) 를 사용하여 키가 없을 때 자동으로 빈 리스트 생성
dept_salaries = defaultdict(list)

# 각 부서별로 급여를 리스트에 추가
for emp in employees:
    dept_salaries[emp["department"]].append(emp["salary"])

# 각 부서별 평균을 딕셔너리 컴프리헨션으로 계산
dept_avg_salary = {dept: sum(salaries) / len(salaries) for dept, salaries in dept_salaries.items()}

print("부서별 평균 급여:", dept_avg_salary)
