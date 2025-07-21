import requests

def get_github_user_info(username: str, token: str = None) -> dict:
    url = f"https://api.github.com/users/{username}"
    headers = {}

    if token:
        headers["Authorization"] = f"token {token}"

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 403:
        raise Exception("❌ 요청이 너무 많아 차단되었습니다. Personal Access Token을 사용해 보세요.")
    elif response.status_code == 404:
        raise ValueError("❌ 존재하지 않는 사용자입니다.")
    else:
        raise Exception(f"❌ 알 수 없는 오류 발생 (status code: {response.status_code})")

def print_user_info(user_info: dict):
    print("\n✅ 사용자 정보:")
    print(f"👤 사용자명 (login): {user_info.get('login')}")
    print(f"🧾 이름 (name): {user_info.get('name')}")
    print(f"📍 위치 (location): {user_info.get('location')}")
    print(f"📖 소개 (bio): {user_info.get('bio')}")
    print(f"🔗 블로그: {user_info.get('blog')}")
    print(f"📅 생성일: {user_info.get('created_at')}")
    print(f"👥 팔로워: {user_info.get('followers')}")
    print(f"👤 팔로잉: {user_info.get('following')}")
    print(f"📦 공개 저장소 수: {user_info.get('public_repos')}")
    print(f"🐙 GitHub 프로필: {user_info.get('html_url')}\n")

def main():
    username = input("GitHub 사용자 이름을 입력하세요: ").strip()
    token = input("Personal Access Token이 있다면 입력하세요 (없으면 Enter): ").strip() or None

    try:
        user_info = get_github_user_info(username, token)
        print_user_info(user_info)
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
