<template>
  <div class="card shadow-sm h-100">
    <!-- 이미지 -->
    <div class="ratio ratio-1x1 bg-light">
      <img
        :src="post.imageUrl"
        :alt="post.caption || 'post image'"
        class="w-100 h-100 object-fit-cover rounded-top"
      />
    </div>

    <div class="card-body">
      <!-- 액션: 좋아요 / 댓글수 -->
      <div class="d-flex align-items-center mb-2">
        <button
          type="button"
          class="btn btn-link p-0 me-3 text-decoration-none"
          :aria-pressed="post.liked.toString()"
          @click="$emit('toggle-like', post.id)"
        >
          <span class="fs-5" :class="post.liked ? 'text-danger' : 'text-body'">
            {{ post.liked ? "♥" : "♡" }}
          </span>
        </button>

        <span class="me-auto small text-muted">
          좋아요 {{ post.likesCount.toLocaleString() }}개
        </span>

        <span class="small text-muted"> 댓글 {{ post.comments.length }} </span>
      </div>

      <!-- 캡션 -->
      <p v-if="post.caption" class="mb-3">
        <strong class="me-1">{{ post.author }}</strong
        >{{ post.caption }}
      </p>

      <!-- 댓글 목록 -->
      <ul class="list-unstyled mb-3">
        <li v-for="(c, i) in visibleComments" :key="i" class="mb-2">
          <strong class="me-1">{{ c.author }}</strong>
          <span>{{ c.text }}</span>
        </li>
      </ul>
      <button
        v-if="canShowMore"
        class="btn btn-sm btn-outline-secondary w-100 mb-3"
        @click="showAll = true"
      >
        댓글 더 보기 ({{ post.comments.length - maxPreview }}개)
      </button>

      <!-- 댓글 입력 -->
      <div class="input-group">
        <input
          v-model.trim="newComment"
          type="text"
          class="form-control"
          placeholder="댓글을 입력하세요"
          @keyup.enter="submitComment"
        />
        <button
          class="btn btn-primary"
          type="button"
          @click="submitComment"
          :disabled="!newComment"
        >
          게시
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, defineProps, defineEmits } from "vue";

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

const props = defineProps<{ post: Post }>();
const emit = defineEmits<{
  (e: "toggle-like", id: number): void;
  (e: "add-comment", payload: { id: number; text: string }): void;
}>();

const newComment = ref("");
const maxPreview = 3;
const showAll = ref(false);

const visibleComments = computed(() =>
  showAll.value ? props.post.comments : props.post.comments.slice(0, maxPreview)
);
const canShowMore = computed(
  () => !showAll.value && props.post.comments.length > maxPreview
);

function submitComment() {
  if (!newComment.value) return;
  emit("add-comment", { id: props.post.id, text: newComment.value });
  newComment.value = "";
}
</script>

<style scoped>
.object-fit-cover {
  object-fit: cover;
}
</style>
