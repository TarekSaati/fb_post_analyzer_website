import React, {useState} from "react";
import {
    Box,
    Button,
    Flex,
    Input,
    InputGroup,
    Modal,
    ModalBody,
    ModalCloseButton,
    ModalContent,
    ModalFooter,
    ModalHeader,
    ModalOverlay,
    Stack,
    Text,
    useDisclosure
} from "@chakra-ui/react";

const PostsContext = React.createContext({
  posts: [], token: '', setPosts: () => {}
})

function Topic() {
  const [topic, setTopic] = React.useState("")
  const {token, setPosts} = React.useContext(PostsContext)

  const handleInput = event  => {
    setTopic(event.target.value)
  }

  const handleSubmit = async () => {
    let options = {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({'topic': topic, 'estimated': true})
  }
    const data = await fetch("https://fastapi-tarek.onrender.com/home/", options)
    const posts = await data.json()
    setPosts(posts)  
  }

  return (
      <InputGroup size="md">
        <Input
          pr="4.5rem"
          type="text"
          placeholder="Select a topic"
          aria-label="Select a topic"
          onChange={handleInput}
        />
        <Button h="2.5rem" size='lg' onClick={handleSubmit}>Go ..</Button>
      </InputGroup>
  )
}

function UpdatePost({oldTopic, id}) {
  const {isOpen, onOpen, onClose} = useDisclosure()
  const [topic, setTopic] = useState(oldTopic)
  const {token, setPosts} = React.useContext(PostsContext)

  const updatePost = async () => {
    let options = {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({'index': id, 'topic': topic})
  }
  await fetch(`https://fastapi-tarek.onrender.com/home/${id}`, options)
  options = {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({'topic': topic, 'estimated': true})
}
  const data = await fetch("https://fastapi-tarek.onrender.com/home/", options)
  const posts = await data.json()
  setPosts(posts)  
  onClose()
    
  }

  return (
    <>
      <Button h="1.5rem" size="sm" onClick={onOpen}>Re-Assign topic</Button>
      <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay/>
        <ModalContent>
          <ModalHeader>Re-Assign topic</ModalHeader>
          <ModalCloseButton/>
          <ModalBody>
            <InputGroup size="md">
              <Input
                pr="4.5rem"
                type="text"
                placeholder="News, Bussiness, Entertainment, Sport, or Education"
                aria-label="News, Bussiness, Entertainment, Sport, Education"
                onChange={e => setTopic(e.target.value)}
              />
            </InputGroup>
          </ModalBody>

          <ModalFooter>
            <Button h="1.5rem" size="sm" onClick={updatePost}>Update Post</Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </>
  )
}

function PostHelper({content, time, pagename, topic, id}) {
  return (
    <Box p={1} shadow="sm">
      <Flex justify="space-between">
      <Stack spacing={2}>
        <Text mt={4} as="div">{pagename}</Text>
        <Text mt={4} as="div">{time}</Text>
        <Text mt={4} as="div">{content}</Text>
        </Stack>
        <Flex align="end">
            <UpdatePost id={id} topic={topic}/>
          </Flex>
      </Flex>
    </Box>
  )
}

export default function Postlist({token}) {
  const [posts, setPosts] = useState([])

  // const fetchPosts = async () => {
  //   let options = {
  //     method: 'GET',
  //     headers: {
  //       'Authorization': `Bearer ${token}`,  
  //       'Content-Type': 'application/json'
  //     },
  // }
  //   const response = await fetch("https://fastapi-tarek.onrender.com/home/", options)
  //   const data = await response.json()
  //   setPosts(data)
  // }
 
  return (
    <PostsContext.Provider value={{posts, token, setPosts}}>
      <Topic />
      <Stack spacing={5}>
        {
          posts.length > 0 && (posts.map((post) => (
            <PostHelper content={post.text} time={post.time} pagename={post.pagename} id={post.index} topic={post.topic} />)
          ))
        }
      </Stack>
    </PostsContext.Provider>
  )
}
