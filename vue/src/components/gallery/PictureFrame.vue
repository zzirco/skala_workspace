<template>
  <slot></slot>
  <h1>액자 ID: {{ frameId }}</h1>
  <slot name="content"></slot>
  <slot name="author"></slot>
</template>

<script setup>
const props = defineProps({
  frameId: String,
});
</script>
