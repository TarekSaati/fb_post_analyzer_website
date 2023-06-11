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
  posts: [], token: '', fetchPosts: () => {}
})

function Topic({token}) {
  const [topic, setTopic] = React.useState("")
  const {fetchPosts} = React.useContext(PostsContext)

  const handleInput = event  => {
    setTopic(event.target.value)
  }

  const handleSubmit = () => {
    const headers = { 'Authorization': {token},
    "Content-Type": "application/json" };
    fetch("http://localhost:8000/home/", {
      method: "POST",
      headers: { headers },
      body: JSON.stringify(topic)
    }).then(fetchPosts)
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
  const {token, fetchPosts} = React.useContext(PostsContext)

  const updatePost = async () => {
    const headers = { 'Authorization': {token},
    "Content-Type": "application/json" };
    await fetch(`http://localhost:8000/home/${id}`, {
      method: "PUT",
      headers: { headers },
      body: JSON.stringify({'index': id, 'topic': topic})
    })
    onClose()
    await fetchPosts()
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
  const fetchPosts = async () => {
    const headers = { 'Authorization': `Bearer ${token}` };
    const response = await fetch("http://localhost:8000/home/", {headers})
    const data = await response.json()
    setPosts(data)
    console.log(token)
  }
 
  return (
    <PostsContext.Provider value={{posts, token, fetchPosts}}>
      <Topic token={token}/>
      <Stack spacing={5}>
        {
          posts.map((post) => (
            <PostHelper content={post.Post.text} time={post.Post.time} pagename={post.Post.pagename} id={post.Post.index} topic={post.Post.topic} />
          ))
        }
      </Stack>
    </PostsContext.Provider>
  )
}
