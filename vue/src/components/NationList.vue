<template>
<div>
<h1>댓글 달기</h1>
<input v-model.trim="comment" placeholder="댓글을 입력하세요" @keyup.enter="addNation"></input>
<button @click="addNation">추가</button>
	<button @click="deleteNation">삭제</button>
		<ol>
				<template v-for="(item, index) in items" :key="index">
						<li>{{ item.comment }}</li>
				</template>
		</ol>
</div>


</template>
<script setup>
import { ref } from 'vue'; 

const comment = ref('')
const items = ref([])

function addNation(event) {
		console.log(event)
		if (comment.value.trim() == '') {
				return
		}
		items.value.push({comment: comment.value})
		comment.value = ''
		console.log(items)
}
function deleteNation() {
		items.value.pop()
		console.log(items)
}
</script>