import re

def 비밀번호_검증(비밀번호):
    오류_메시지 = []

    if not re.search(r"[a-z]", 비밀번호):
        오류_메시지.append("❌ 소문자가 포함되어야 합니다.")
    if not re.search(r"[A-Z]", 비밀번호):
        오류_메시지.append("❌ 대문자가 포함되어야 합니다.")
    if not re.search(r"[0-9]", 비밀번호):
        오류_메시지.append("❌ 숫자가 포함되어야 합니다.")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>_\-+=\\[\]/~`']", 비밀번호):
        오류_메시지.append("❌ 특수 기호가 포함되어야 합니다.")

    return 오류_메시지

def main():
    while True:
        입력값 = input("비밀번호를 입력하세요 (!quit 입력 시 종료): ")
        if 입력값 == "!quit":
            print("프로그램을 종료합니다.")
            break

        오류들 = 비밀번호_검증(입력값)
        if not 오류들:
            print("✅ 유효한 비밀번호입니다.")
        else:
            print("❌ 유효하지 않은 비밀번호입니다. 다음 조건들을 확인하세요:")
            for 오류 in 오류들:
                print(오류)

if __name__ == "__main__":
    main()
