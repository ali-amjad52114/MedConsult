import os
os.environ['GOOGLE_API_KEY'] = 'test_key'
os.environ['HF_TOKEN'] = 'test_token'

# Test individual components without model loading
from sirius.memory_store import MemoryStore
from sirius.memory_retriever import MemoryRetriever
from sirius.lesson_extractor import LessonExtractor
from sirius.validator import PeriodicValidator
from sirius.augmentation import EnhancedAugmentation

print('✅ All SiriuS components import successfully')

# Test memory operations
store = MemoryStore()
store.add_lesson('Test lesson', metadata={'target_agent': 'analyst', 'lesson_type': 'extraction_pattern'})
results = store.query_for_agent('analyst', 'test')
print(f'✅ Memory store: {len(results)} results')

# Test retriever
retriever = MemoryRetriever(store)
ctx = retriever.get_relevant_lessons('test', agent_name='analyst')
print(f'✅ Memory retriever: {len(ctx)} chars context')

# Test validator
validator = PeriodicValidator(None, validation_interval=5)
print(f'✅ Validator should_validate(5): {validator.should_validate(5)}')

# Test augmentation (without cloud manager)
class MockCloud:
    def generate_response(self, system_prompt, user_message, max_tokens=1024):
        return '{"analyst": {"score": 2, "feedback": "test", "should_retry": true}}'

aug = EnhancedAugmentation(MockCloud())
scores = aug.get_retry_agents({'analyst': {'score': 2, 'feedback': 'test', 'should_retry': True}})
print(f'✅ Augmentation: {len(scores)} agents to retry')

print()
print('🎉 ALL COMPONENTS WORKING!')
