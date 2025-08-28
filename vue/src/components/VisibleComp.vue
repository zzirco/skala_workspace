<template>
  <template v-if="isVisible">
    <div class="base" id="1">V-IF TRUE 일때 보입니다.</div>
    <p>나도 v-if일떄 보여야 하는데</p>
  </template>
  <template v-else>
    <div class="base" id="2">V-IF FALSE 일때 보입니다.</div>
  </template>
  <div class="base" id="3" v-show="isVisible">V-SHOW TRUE 일때 보입니다.</div>
</template>

<script setup>
import { ref } from "vue";
const isVisible = ref(true);
</script>
<style>
.base {
  margin: 2px;
  border: 1px solid;
  height: 30px;
}
</style>
