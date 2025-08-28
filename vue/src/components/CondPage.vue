<template>
  <a :href="link">내 페이지</a>
</template>
<script setup>
const isLoggedIn = false;
const link = isLoggedIn ? "https://naver.com" : "https://google.com";
</script>
