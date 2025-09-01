<template>
  <div class="container py-4">
    <div class="d-flex align-items-center justify-content-between mb-3">
      <h1 class="h4 mb-0">피드</h1>
      <div class="text-muted small">데모 데이터</div>
    </div>

    <!-- 카드 그리드: 1열(md전), 2열(md~), 3열(lg~) -->
    <div class="row g-4">
      <div
        v-for="post in posts"
        :key="post.id"
        class="col-12 col-md-6 col-lg-4 d-flex"
      >
        <PostCard
          class="w-100"
          :post="post"
          @toggle-like="toggleLike"
          @add-comment="addComment"
        />
      </div>
    </div>

    <!-- 더보기(페이징/무한스크롤 훅 자리) -->
    <div class="d-grid mt-4">
      <button class="btn btn-outline-secondary" type="button" @click="loadMore">
        더 보기
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PostCard from "./PostCard.vue";

type Comment = { author: string; text: string };
type Post = {
  id: number;
  author: string;
  imageUrl: string;
  caption?: string;
  liked: boolean;
  likesCount: number;
  comments: Comment[];
};

// 데모 데이터
const posts = ref<Post[]>([
  {
    id: 1,
    author: "hojun",
    imageUrl:
      "https://images.unsplash.com/photo-1517486808906-6ca8b3f04846?q=80&w=1200&auto=format&fit=crop",
    caption: "주말 등산 ☀️",
    liked: false,
    likesCount: 128,
    comments: [
      { author: "dev_j", text: "뷰 미쳤다!" },
      { author: "maria", text: "공기 좋아보여요" },
      { author: "lee", text: "어디 산인가요?" },
      { author: "kim", text: "다음엔 같이 가요!" },
    ],
  },
  {
    id: 2,
    author: "skala",
    imageUrl:
      "https://images.unsplash.com/photo-1491553895911-0055eca6402d?q=80&w=1200&auto=format&fit=crop",
    caption: "커피타임 ☕",
    liked: true,
    likesCount: 432,
    comments: [
      { author: "eun", text: "아메리카노가 진리" },
      { author: "park", text: "라떼도 좋아요" },
    ],
  },
  {
    id: 3,
    author: "akify",
    imageUrl:
      "https://images.unsplash.com/photo-1520975916090-3105956dac38?q=80&w=1200&auto=format&fit=crop",
    caption: "밤의 도시 산책",
    liked: false,
    likesCount: 76,
    comments: [],
  },
]);

function toggleLike(id: number) {
  const p = posts.value.find((x) => x.id === id);
  if (!p) return;
  p.liked = !p.liked;
  p.likesCount += p.liked ? 1 : -1;
}

function addComment({ id, text }: { id: number; text: string }) {
  const p = posts.value.find((x) => x.id === id);
  if (!p) return;
  p.comments.push({ author: "you", text });
}

function loadMore() {
  // TODO: 실제 API 연동 시 다음 페이지 불러오기
  const nextId = posts.value.length + 1;
  posts.value.push({
    id: nextId,
    author: "guest",
    imageUrl:
      "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?q=80&w=1200&auto=format&fit=crop",
    caption: "새 포스트 예시",
    liked: false,
    likesCount: 0,
    comments: [],
  });
}
</script>
