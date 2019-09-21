from collections.abc import Sequence

class InstList(Sequence):
    '''
    Structure for applying functions to many instances
    
    This class is a holder for many instances of the same 
    type of object, self.obj_type. InstList behaves like a 
    list, so indexing into it, e.g. via
    
        my_inst_list = InstList(list_of_instances, obj_type = Image)
        my_inst_list[i]
    
    will return the instance in list_of_instances[i].
    
    It contains functionality to apply an augmentation 
    function to all instances at once via its method 
    `apply_func`. The functions applied are assumed to 
    accept one required argument, an instance of 
    self.obj_type. They are also assumed to be operated 
    via optional keyword arguments, which can be passed 
    to `apply_func`.
    

    Example workflow:
        ```
        import audio_aug as aa
        
        # Load audio file
        audio_file_path = '../tests/1min.wav'
        audio = aa.Audio(label = 'test', path = audio_file_path)

        # Make a list of 10 audio chunks from this file
        from copy import deepcopy
        audios = []
        for _ in range(10):
            audios.append(deepcopy(audio))

        # Create InstList object from the list of 10 chunks
        audio_list = InstList(instances = audios, obj_type = aa.Audio)

        # Apply many manipulation functions, passing keyword arguments
        audio_list = audio_list \\
            .apply_func(
                func = aa.get_chunk,
                chance_random_skip = 1) \\
            .apply_func(func = aa.cyclic_shift) \\
            .apply_func(
                func = aa.time_stretch_divisions,
                low_division_duration = 0.1,
                high_division_duration = 3) \\
            .apply_func(func = aa.pitch_shift_divisions) \\
            .apply_func(func = aa.random_filter) \\
            .apply_func(func = aa.sum_chunks)
        ```
    '''
    
    def __init__(self, instances, obj_type):
        '''
        Create InstList
        
        Args:
            instances (list): list of instances
            obj_type (class): type that each instance should be
        '''
        
        if not isinstance(instances, list):
            raise ValueError('instances is not list')
        for inst in instances:
            if not isinstance(inst, obj_type):
                raise ValueError('all elements of audios must be instance of Audio class')
        
        self.obj_type = obj_type
        self.instances = instances
        super().__init__()
        
    def __getitem__(self, i):
        return self.instances[i]
    
    def __len__(self):
        return len(self.instances)
        
    
    def apply_func(self, func, **kwargs):
        '''
        Apply a function to all instances in self.instances
        
        Args:
            func (callable): function that operates on objects
                of type self.obj_type
            **kwargs: kwargs for function
            
        Returns:
            self, for dotchaining
        '''
        
        if not callable(func):
            raise ValueError('func is not callable')

        for idx, inst in enumerate(self.instances):
            self.instances[idx] = func(inst, **kwargs)
            
        # For dotchaining
        return self